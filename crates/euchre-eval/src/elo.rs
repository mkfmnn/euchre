//! Batch Elo ratings from round-robin results, the BayesElo way.
//!
//! Incremental Elo (update after every game) is order-dependent and noisy when
//! ranking a fixed pool of agents. This module instead fits all ratings at once
//! by maximum likelihood under the **Bradley-Terry** model — the model whose
//! win probability *is* the Elo formula:
//!
//! ```text
//! P(i beats j) = 1 / (1 + 10^(-(r_i - r_j) / 400))
//! ```
//!
//! Given the whole win matrix, [`fit`] finds the ratings that make the observed
//! results most likely. Two refinements make it robust:
//!
//! * A small **Bayesian prior** — each agent plays a few virtual, evenly-split
//!   games against a fixed rating-0 anchor. This keeps an agent that sweeps every
//!   game at a finite rating, connects the comparison graph, and pins the
//!   otherwise-floating scale. It is the "Bayes" in BayesElo.
//! * An **uncertainty** for each rating, the standard error read off the inverse
//!   Fisher information of the fit — wide for agents with few or lopsided games,
//!   tight for those with many close ones.
//!
//! The optimiser is Hunter's MM algorithm (a guaranteed-to-converge fixed-point
//! iteration for Bradley-Terry); the standard errors come from inverting the
//! information matrix. No external solver, and both are checked against closed
//! forms in the tests.

/// Elo points per natural log-unit of Bradley-Terry strength: `400 / ln 10`.
///
/// A rating `r` corresponds to strength `γ = 10^(r / 400) = exp(r / ELO_SCALE)`,
/// so a difference in natural log-strength scales to Elo by this factor.
const ELO_SCALE: f64 = 400.0 / std::f64::consts::LN_10;

/// One agent's fitted rating.
#[derive(Debug, Clone, PartialEq)]
pub struct Rating {
    /// The agent's name, carried through from the input.
    pub name: String,
    /// The fitted Elo rating, recentred so the pool averages 0.
    pub elo: f64,
    /// The approximate standard error of [`elo`](Self::elo), from the inverse
    /// Fisher information of the fit. Roughly a one-sigma uncertainty band.
    pub elo_stderr: f64,
    /// Total match wins across every opponent.
    pub wins: u64,
    /// Total match losses across every opponent.
    pub losses: u64,
}

impl Rating {
    /// Total games played (`wins + losses`).
    pub fn games(&self) -> u64 {
        self.wins + self.losses
    }

    /// The raw win rate, or `0.0` if the agent played no games.
    pub fn win_rate(&self) -> f64 {
        if self.games() == 0 {
            0.0
        } else {
            self.wins as f64 / self.games() as f64
        }
    }
}

/// Knobs for the [`fit`].
#[derive(Debug, Clone, Copy)]
pub struct EloOptions {
    /// Strength of the Bayesian prior, in virtual games each agent plays against
    /// a fixed rating-0 anchor (winning exactly half). A small positive value
    /// keeps sweepers finite and the scale pinned without materially biasing a
    /// fit backed by hundreds of real games. Must be `> 0`.
    pub prior_games: f64,
    /// Maximum MM iterations before giving up on the convergence tolerance.
    pub max_iters: usize,
    /// Convergence tolerance: stop once no strength changes by more than this
    /// fraction in an iteration.
    pub tolerance: f64,
}

impl Default for EloOptions {
    fn default() -> Self {
        EloOptions {
            prior_games: 2.0,
            max_iters: 10_000,
            tolerance: 1e-10,
        }
    }
}

/// Fits Elo ratings to a square win matrix by Bradley-Terry maximum likelihood.
///
/// `wins[i][j]` is the number of games agent `i` won against agent `j` (the
/// diagonal is ignored). `names[i]` labels row/column `i`. The returned ratings
/// are in input order — recentred so the pool's mean rating is 0 — each carrying
/// its standard error and aggregate win/loss totals.
///
/// # Panics
///
/// Panics if `wins` is not square with side `names.len()`, or if
/// `opts.prior_games` is not positive (the prior is what guarantees a unique,
/// finite fit).
pub fn fit(names: &[String], wins: &[Vec<u64>], opts: &EloOptions) -> Vec<Rating> {
    let n = names.len();
    assert_eq!(wins.len(), n, "win matrix must have one row per agent");
    assert!(
        wins.iter().all(|row| row.len() == n),
        "win matrix must be square"
    );
    assert!(opts.prior_games > 0.0, "prior_games must be positive");

    // Total games between each ordered pair, and each agent's win/loss totals.
    let mut total_wins = vec![0u64; n];
    let mut total_losses = vec![0u64; n];
    let mut games = vec![vec![0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let n_ij = wins[i][j] + wins[j][i];
            games[i][j] = n_ij as f64;
            total_wins[i] += wins[i][j];
            total_losses[i] += wins[j][i];
        }
    }

    let gamma = solve_mm(wins, &games, opts);

    // Convert strengths to Elo and recentre so the pool averages 0.
    let mut elo: Vec<f64> = gamma.iter().map(|g| ELO_SCALE * g.ln()).collect();
    if n > 0 {
        let mean = elo.iter().sum::<f64>() / n as f64;
        for e in &mut elo {
            *e -= mean;
        }
    }

    let stderr = standard_errors(&gamma, &games, opts.prior_games);

    (0..n)
        .map(|i| Rating {
            name: names[i].clone(),
            elo: elo[i],
            elo_stderr: stderr[i],
            wins: total_wins[i],
            losses: total_losses[i],
        })
        .collect()
}

/// Sorts ratings into leaderboard order: strongest first, ties broken by name.
pub fn leaderboard(mut ratings: Vec<Rating>) -> Vec<Rating> {
    ratings.sort_by(|a, b| {
        b.elo
            .partial_cmp(&a.elo)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });
    ratings
}

/// Runs Hunter's MM iteration to the Bradley-Terry MAP strengths `γ`.
///
/// Each step replaces `γ_i` by its wins (real, plus half the virtual anchor
/// games) divided by `Σ_j n_ij / (γ_i + γ_j)` over all opponents including the
/// fixed-strength anchor. The update never decreases the penalised likelihood
/// and converges to its unique maximiser.
fn solve_mm(wins: &[Vec<u64>], games: &[Vec<f64>], opts: &EloOptions) -> Vec<f64> {
    let n = wins.len();
    let prior = opts.prior_games;
    let mut gamma = vec![1.0f64; n];

    for _ in 0..opts.max_iters {
        let mut max_rel_change = 0.0f64;
        for i in 0..n {
            // Wins of i: real wins plus half its virtual games against the anchor.
            let mut numerator = prior / 2.0;
            for (j, &w) in wins[i].iter().enumerate() {
                if i != j {
                    numerator += w as f64;
                }
            }
            // Expected-loss denominator over real opponents and the anchor (γ=1).
            let mut denominator = prior / (gamma[i] + 1.0);
            for j in 0..n {
                if i != j && games[i][j] > 0.0 {
                    denominator += games[i][j] / (gamma[i] + gamma[j]);
                }
            }
            let next = numerator / denominator;
            let rel = ((next - gamma[i]) / gamma[i]).abs();
            max_rel_change = max_rel_change.max(rel);
            gamma[i] = next;
        }
        if max_rel_change < opts.tolerance {
            break;
        }
    }
    gamma
}

/// Approximate Elo standard errors, measured relative to the pool mean.
///
/// The ratings are reported recentred on the field, so the uncertainty that
/// matters is each rating's *relative* to that mean, not its absolute level —
/// which a light prior barely pins, and which would otherwise swamp every error
/// bar with the same large shared term. This applies the centring contrast `P =
/// I - J/n` to the parameter [`covariance`] (`diag(P C P)`) before scaling to
/// Elo. Returns `f64::INFINITY` for any agent whose information is degenerate.
fn standard_errors(gamma: &[f64], games: &[Vec<f64>], prior: f64) -> Vec<f64> {
    let n = gamma.len();
    let Some(cov) = covariance(gamma, games, prior) else {
        return vec![f64::INFINITY; n];
    };
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    // Grand mean of all covariance entries, the J C J / n^2 term shared by every
    // centred variance.
    let grand: f64 = cov.iter().flatten().sum::<f64>() / (nf * nf);
    (0..n)
        .map(|i| {
            // Row mean of the covariance, the (J C)_i / n cross term for agent i.
            let row_mean: f64 = cov[i].iter().sum::<f64>() / nf;
            let var = cov[i][i] - 2.0 * row_mean + grand;
            if var > 0.0 {
                ELO_SCALE * var.sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect()
}

/// The parameter covariance of the fit, in natural-log-strength space.
///
/// In that space the negative log-likelihood's Hessian is the
/// graph-Laplacian-like Fisher information matrix `H`, with `H_ii = Σ_j n_ij
/// p_ij(1 - p_ij)` and `H_ij = -n_ij p_ij(1 - p_ij)`; the prior adds each
/// agent's anchor games to the diagonal, which is what makes the otherwise
/// translation-singular `H` invertible. The covariance is `H^{-1}`. The variance
/// of an individual rating depends on how firmly the prior pins the overall
/// level, but the variance of a *difference* `Var(θ_i) + Var(θ_j) - 2 Cov(θ_i,
/// θ_j)` is essentially prior-free. Returns `None` if `H` is singular.
fn covariance(gamma: &[f64], games: &[Vec<f64>], prior: f64) -> Option<Vec<Vec<f64>>> {
    let n = gamma.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let mut h = vec![vec![0f64; n]; n];
    for i in 0..n {
        // Anchor contribution: prior games against the fixed γ=1 opponent.
        let p_anchor = gamma[i] / (gamma[i] + 1.0);
        h[i][i] += prior * p_anchor * (1.0 - p_anchor);
        for j in 0..n {
            if i == j || games[i][j] == 0.0 {
                continue;
            }
            let p = gamma[i] / (gamma[i] + gamma[j]);
            let info = games[i][j] * p * (1.0 - p);
            h[i][i] += info;
            h[i][j] -= info;
        }
    }
    invert(&h)
}

/// Inverts a small square matrix by Gauss-Jordan elimination with partial
/// pivoting, or returns `None` if it is singular. Sized for a handful of agents.
fn invert(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    // Augment [A | I].
    let mut a: Vec<Vec<f64>> = matrix
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivot: swap in the row with the largest magnitude in this column.
        let pivot = (col..n).max_by(|&r1, &r2| {
            a[r1][col]
                .abs()
                .partial_cmp(&a[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);

        let diag = a[col][col];
        for x in a[col].iter_mut() {
            *x /= diag;
        }
        // Eliminate this column from every other row. The pivot row is fixed for
        // the sweep, so snapshot it once and subtract a multiple from each row.
        let pivot_row = a[col].clone();
        for (row, target) in a.iter_mut().enumerate() {
            if row == col {
                continue;
            }
            let factor = target[col];
            if factor != 0.0 {
                for (dst, &src) in target.iter_mut().zip(pivot_row.iter()) {
                    *dst -= factor * src;
                }
            }
        }
    }

    Some(a.iter().map(|row| row[n..].to_vec()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("a{i}")).collect()
    }

    #[test]
    fn two_player_difference_matches_bradley_terry() {
        // With a negligible prior the model reduces to pure Bradley-Terry, where
        // the Elo gap between two players is 400 * log10(wins_a / wins_b).
        let wins = vec![vec![0, 75], vec![25, 0]];
        let opts = EloOptions {
            prior_games: 1e-6,
            ..EloOptions::default()
        };
        let r = fit(&names(2), &wins, &opts);
        let diff = r[0].elo - r[1].elo;
        let expected = 400.0 * 3f64.log10(); // 75/25 = 3
        assert!(
            (diff - expected).abs() < 0.5,
            "diff = {diff}, want {expected}"
        );
        // Recentred symmetrically around zero.
        assert!((r[0].elo + r[1].elo).abs() < 1e-6);
    }

    #[test]
    fn two_player_difference_variance_matches_closed_form() {
        // For a 2-player Bradley-Terry fit the variance of the rating *difference*
        // is (a + b) / (a * b) in log space (the prior-free, well-conditioned
        // quantity; individual variances depend on how the prior pins the level).
        // This exercises the Hessian build and the general matrix inverse against
        // a closed form.
        let (a, b) = (60u64, 40u64);
        let games = vec![vec![0.0, (a + b) as f64], vec![(a + b) as f64, 0.0]];
        let opts = EloOptions {
            prior_games: 1e-4,
            ..EloOptions::default()
        };
        let wins = vec![vec![0, a], vec![b, 0]];
        let gamma = solve_mm(&wins, &games, &opts);
        let cov = covariance(&gamma, &games, opts.prior_games).unwrap();
        let var_diff = cov[0][0] + cov[1][1] - 2.0 * cov[0][1];
        let expected = (a + b) as f64 / (a * b) as f64;
        assert!(
            (var_diff - expected).abs() < expected * 0.01,
            "var_diff = {var_diff}, want {expected}"
        );
    }

    #[test]
    fn more_games_shrink_the_error_bar() {
        // Marginal standard errors must tighten as the same matchup is played more.
        let opts = EloOptions::default();
        let few = fit(&names(2), &[vec![0, 30], vec![20, 0]], &opts);
        let many = fit(&names(2), &[vec![0, 300], vec![200, 0]], &opts);
        assert!(
            many[0].elo_stderr < few[0].elo_stderr,
            "{} !< {}",
            many[0].elo_stderr,
            few[0].elo_stderr
        );
    }

    #[test]
    fn transitive_results_are_ordered() {
        // a0 > a1 > a2 in head-to-heads should yield strictly decreasing ratings.
        let wins = vec![vec![0, 70, 90], vec![30, 0, 65], vec![10, 35, 0]];
        let r = fit(&names(3), &wins, &EloOptions::default());
        assert!(r[0].elo > r[1].elo, "{} !> {}", r[0].elo, r[1].elo);
        assert!(r[1].elo > r[2].elo, "{} !> {}", r[1].elo, r[2].elo);
    }

    #[test]
    fn a_sweeper_is_finite_top_and_uncertain() {
        // An agent that wins every game must still get a finite rating (the prior),
        // the highest one, and a wider error bar than a well-measured mid agent.
        let wins = vec![vec![0, 100, 100], vec![0, 0, 50], vec![0, 50, 0]];
        let r = fit(&names(3), &wins, &EloOptions::default());
        assert!(r[0].elo.is_finite());
        assert!(r[0].elo > r[1].elo && r[0].elo > r[2].elo);
        assert!(r[0].elo_stderr.is_finite());
        assert!(r[0].elo_stderr > r[1].elo_stderr);
    }

    #[test]
    fn identical_records_give_equal_ratings() {
        // Two agents with mirror-image records sit at the same rating.
        let wins = vec![vec![0, 50, 60], vec![50, 0, 60], vec![40, 40, 0]];
        let r = fit(&names(3), &wins, &EloOptions::default());
        assert!(
            (r[0].elo - r[1].elo).abs() < 1e-6,
            "{} vs {}",
            r[0].elo,
            r[1].elo
        );
    }

    #[test]
    fn leaderboard_sorts_strongest_first() {
        let wins = vec![vec![0, 30, 90], vec![70, 0, 80], vec![10, 20, 0]];
        let board = leaderboard(fit(&names(3), &wins, &EloOptions::default()));
        assert!(board[0].elo >= board[1].elo && board[1].elo >= board[2].elo);
    }

    #[test]
    fn win_and_loss_totals_are_aggregated() {
        let wins = vec![vec![0, 30, 90], vec![70, 0, 80], vec![10, 20, 0]];
        let r = fit(&names(3), &wins, &EloOptions::default());
        assert_eq!(r[0].wins, 120);
        assert_eq!(r[0].losses, 80);
        assert_eq!(r[0].games(), 200);
        assert!((r[0].win_rate() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn invert_matches_closed_form() {
        // det([[4,1],[2,3]]) = 10, so the inverse is [[3,-1],[-2,4]] / 10.
        let m = vec![vec![4.0, 1.0], vec![2.0, 3.0]];
        let inv = invert(&m).unwrap();
        let want = [[0.3, -0.1], [-0.2, 0.4]];
        for (got_row, want_row) in inv.iter().zip(want.iter()) {
            for (got, want) in got_row.iter().zip(want_row.iter()) {
                assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
            }
        }
    }
}
