//! Statistics for comparing agents on a **win-probability** objective.
//!
//! The metric that matters in Euchre is *how often an agent wins the match*,
//! not how many points it piles up — an agent that correctly sandbags a lead or
//! gambles from behind trades expected points for win probability on purpose, so
//! a points-based score would penalise good play. Everything here therefore
//! works in the currency of match wins:
//!
//! * [`wilson_interval`] — a confidence interval for a raw win rate that behaves
//!   near 0/1 and for small samples, where the textbook Wald interval does not.
//! * [`mcnemar`] — the paired test for the duplicate-dealing design, where each
//!   deal is played from both sides so shared deal luck cancels.
//! * [`Sprt`] — a sequential test that stops as soon as the data decide whether
//!   one agent is meaningfully stronger, spending only as many games as needed.

/// The `z` multiplier for a two-sided 95% confidence interval.
pub const Z_95: f64 = 1.959_963_984_540_054;

/// A Wilson score interval for a binomial proportion.
///
/// Given `successes` out of `n` trials, returns the lower and upper bounds of
/// the `z`-level confidence interval for the true success probability. Unlike
/// the normal (Wald) approximation it is always within `[0, 1]` and keeps its
/// nominal coverage for lopsided results and small `n`. Pass [`Z_95`] for the
/// usual 95% interval.
///
/// An empty sample (`n == 0`) returns the maximally uninformative `(0.0, 1.0)`.
pub fn wilson_interval(successes: u64, n: u64, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let nf = n as f64;
    let p = successes as f64 / nf;
    let z2 = z * z;
    let denom = 1.0 + z2 / nf;
    let center = (p + z2 / (2.0 * nf)) / denom;
    let half = (z / denom) * (p * (1.0 - p) / nf + z2 / (4.0 * nf * nf)).sqrt();
    ((center - half).max(0.0), (center + half).min(1.0))
}

/// The outcome of a [`mcnemar`] test on paired binary results.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McNemar {
    /// Pairs where the first agent won and the second lost (favouring the first).
    pub a_better: u64,
    /// Pairs where the second agent won and the first lost (favouring the second).
    pub b_better: u64,
    /// Two-sided p-value for the null hypothesis that the two agents are equal.
    pub p_value: f64,
}

/// McNemar's test for paired binary data.
///
/// In the duplicate-dealing design every deal is played twice with the agents on
/// opposite sides, so each deal yields a matched pair. `a_better` counts the
/// deals decided in the first agent's favour once the cards are held fixed, and
/// `b_better` the reverse; deals that came out the same either way (the cards, not
/// the play, decided them) carry no information and are excluded.
///
/// The p-value is an exact two-sided sign test on the discordant pairs, falling
/// back to a continuity-corrected normal approximation only when their count is
/// too large to sum exactly.
pub fn mcnemar(a_better: u64, b_better: u64) -> McNemar {
    let n = a_better + b_better;
    let p_value = if n == 0 {
        1.0
    } else if n <= 5_000_000 {
        sign_test_exact(a_better.min(b_better), n)
    } else {
        // Continuity-corrected normal approximation of the chi-square statistic.
        let diff = (a_better as f64 - b_better as f64).abs();
        let chi2 = (diff - 1.0).max(0.0).powi(2) / n as f64;
        chi_square_1df_sf(chi2)
    };
    McNemar {
        a_better,
        b_better,
        p_value,
    }
}

/// Two-sided p-value of a sign test: `P(X <= k) * 2`, clamped to 1, for
/// `X ~ Binomial(n, 0.5)` with `k <= n / 2`.
fn sign_test_exact(k: u64, n: u64) -> f64 {
    let ln_half = 0.5_f64.ln();
    let mut tail = 0.0;
    for i in 0..=k {
        tail += (ln_choose(n, i) + n as f64 * ln_half).exp();
    }
    (2.0 * tail).min(1.0)
}

/// `ln(n choose k)` via log-gamma, stable for large `n`.
fn ln_choose(n: u64, k: u64) -> f64 {
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

/// Lanczos approximation of `ln(gamma(x))`, valid for `x >= 0.5` (all callers
/// here pass integer arguments `>= 1`).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let t = x + G + 0.5;
    let mut a = COEF[0];
    for (i, c) in COEF.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Survival function of a chi-square distribution with one degree of freedom:
/// `P(X > chi2) = erfc(sqrt(chi2 / 2))`.
fn chi_square_1df_sf(chi2: f64) -> f64 {
    erfc((chi2 / 2.0).sqrt())
}

/// Complementary error function, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
fn erfc(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
        + 0.254_829_592)
        * t
        * (-x * x).exp();
    // erf(x) = sign * (1 - y); erfc = 1 - erf.
    1.0 - sign * (1.0 - y)
}

/// Converts an Elo difference into the win probability it implies.
pub fn elo_to_win_prob(elo: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf(-elo / 400.0))
}

/// What a [`Sprt`] concludes from the games seen so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprtVerdict {
    /// Accept H1: the first agent is at least the H1 strength stronger.
    AcceptH1,
    /// Accept H0: the first agent is not meaningfully stronger.
    AcceptH0,
    /// Not enough evidence yet; keep playing.
    Continue,
}

/// A sequential probability ratio test on a win/loss stream.
///
/// Borrowed from computer-chess engine testing: rather than fixing the number of
/// games up front, it tests H0 (`win_prob = p0`) against H1 (`win_prob = p1`)
/// after every game and stops the moment the evidence crosses a bound, spending
/// only as many games as the effect size warrants. Specify the hypotheses in Elo
/// — e.g. `p0 = 0`, `p1 = 10` asks "is the change worth at least ~10 Elo?".
///
/// Draws do not occur in Euchre, so each game is a clean win or loss.
#[derive(Debug, Clone, Copy)]
pub struct Sprt {
    p0: f64,
    p1: f64,
    lower: f64,
    upper: f64,
}

impl Sprt {
    /// Builds a test of `H0: win_prob = p0` against `H1: win_prob = p1` with the
    /// given type-I (`alpha`) and type-II (`beta`) error rates.
    pub fn new(p0: f64, p1: f64, alpha: f64, beta: f64) -> Self {
        Sprt {
            p0,
            p1,
            lower: (beta / (1.0 - alpha)).ln(),
            upper: ((1.0 - beta) / alpha).ln(),
        }
    }

    /// Convenience constructor taking the hypotheses as Elo gains.
    pub fn from_elo(elo0: f64, elo1: f64, alpha: f64, beta: f64) -> Self {
        Sprt::new(elo_to_win_prob(elo0), elo_to_win_prob(elo1), alpha, beta)
    }

    /// The log-likelihood ratio after `wins` wins and `losses` losses.
    pub fn llr(&self, wins: u64, losses: u64) -> f64 {
        wins as f64 * (self.p1 / self.p0).ln()
            + losses as f64 * ((1.0 - self.p1) / (1.0 - self.p0)).ln()
    }

    /// The lower (accept H0) and upper (accept H1) decision bounds on the LLR.
    pub fn bounds(&self) -> (f64, f64) {
        (self.lower, self.upper)
    }

    /// The current verdict given the running win/loss counts.
    pub fn verdict(&self, wins: u64, losses: u64) -> SprtVerdict {
        let llr = self.llr(wins, losses);
        if llr >= self.upper {
            SprtVerdict::AcceptH1
        } else if llr <= self.lower {
            SprtVerdict::AcceptH0
        } else {
            SprtVerdict::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_brackets_the_point_estimate() {
        let (lo, hi) = wilson_interval(60, 100, Z_95);
        assert!(lo < 0.60 && 0.60 < hi);
        // Known value for 60/100 at 95%: roughly (0.502, 0.691).
        assert!((lo - 0.502).abs() < 0.005, "lo = {lo}");
        assert!((hi - 0.691).abs() < 0.005, "hi = {hi}");
    }

    #[test]
    fn wilson_stays_in_unit_interval_at_extremes() {
        let (lo, hi) = wilson_interval(0, 5, Z_95);
        assert!(lo >= 0.0 && hi <= 1.0);
        let (lo, hi) = wilson_interval(5, 5, Z_95);
        assert!(lo >= 0.0 && hi <= 1.0 && lo > 0.0);
    }

    #[test]
    fn mcnemar_balanced_is_insignificant() {
        let r = mcnemar(50, 50);
        assert!(r.p_value > 0.9, "p = {}", r.p_value);
    }

    #[test]
    fn mcnemar_lopsided_is_significant() {
        let r = mcnemar(40, 10);
        assert!(r.p_value < 0.001, "p = {}", r.p_value);
    }

    #[test]
    fn mcnemar_no_discordant_pairs_is_undecided() {
        assert_eq!(mcnemar(0, 0).p_value, 1.0);
    }

    #[test]
    fn ln_gamma_matches_factorials() {
        // ln_gamma(n+1) == ln(n!).
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-9);
        assert!((ln_gamma(6.0) - 120f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn sprt_accepts_h1_on_a_strong_lead() {
        let sprt = Sprt::from_elo(0.0, 50.0, 0.05, 0.05);
        assert_eq!(sprt.verdict(0, 0), SprtVerdict::Continue);
        // A heavy win rate should eventually clear the H1 bound.
        assert_eq!(sprt.verdict(400, 100), SprtVerdict::AcceptH1);
    }

    #[test]
    fn sprt_accepts_h0_on_a_coin_flip() {
        let sprt = Sprt::from_elo(0.0, 50.0, 0.05, 0.05);
        assert_eq!(sprt.verdict(500, 500), SprtVerdict::AcceptH0);
    }
}
