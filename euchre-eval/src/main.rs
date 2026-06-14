//! Command-line head-to-head runner for Euchre agents.
//!
//! ```text
//! euchre-eval <agent_a> <agent_b> [options]
//!
//!   --games N          number of matches to play (rounded down to an even
//!                      number of duplicate pairs); default 2000
//!   --seed S           base deck seed; default 0
//!   --target-score T   score a team must reach to win a match; default 10
//!   --sprt             stop early once the result is decided (sequential test)
//!   --elo0 E           H0 Elo gain for SPRT; default 0
//!   --elo1 E           H1 Elo gain for SPRT; default 10
//!   --alpha A          SPRT type-I error rate; default 0.05
//!   --beta B           SPRT type-II error rate; default 0.05
//! ```
//!
//! In fixed-games mode it reports the first agent's win rate with a Wilson 95%
//! interval and McNemar's paired p-value. With `--sprt` it plays duplicate pairs
//! until the sequential test accepts H0 or H1 (or `--games` is exhausted).

use std::process::ExitCode;

use euchre_engine::GameConfig;
use euchre_eval::runner::{HeadToHead, run_pair};
use euchre_eval::stats::{Sprt, SprtVerdict, Z_95, mcnemar, wilson_interval};
use euchre_eval::{BUILTIN_AGENTS, builtin};

/// Parsed command-line options.
struct Options {
    agent_a: String,
    agent_b: String,
    games: u64,
    seed: u64,
    target_score: u8,
    sprt: bool,
    elo0: f64,
    elo1: f64,
    alpha: f64,
    beta: f64,
}

fn main() -> ExitCode {
    let opts = match parse_args() {
        Ok(opts) => opts,
        Err(msg) => {
            eprintln!("{msg}\n");
            eprintln!("usage: euchre-eval <agent_a> <agent_b> [options]");
            eprintln!("known agents: {}", BUILTIN_AGENTS.join(", "));
            return ExitCode::FAILURE;
        }
    };

    let Some(a) = builtin(&opts.agent_a) else {
        eprintln!("unknown agent: {}", opts.agent_a);
        return ExitCode::FAILURE;
    };
    let Some(b) = builtin(&opts.agent_b) else {
        eprintln!("unknown agent: {}", opts.agent_b);
        return ExitCode::FAILURE;
    };

    let config = GameConfig {
        target_score: opts.target_score,
        ..GameConfig::default()
    };

    println!(
        "{} (A) vs {} (B) — to {} points, duplicate dealing\n",
        a.name, b.name, opts.target_score
    );

    if opts.sprt {
        run_sprt_mode(&opts, config, &a, &b);
    } else {
        run_fixed_mode(&opts, config, &a, &b);
    }
    ExitCode::SUCCESS
}

/// Plays a fixed number of pairs and prints win rate, Wilson interval and McNemar.
fn run_fixed_mode(
    opts: &Options,
    config: GameConfig,
    a: &euchre_eval::runner::Contestant,
    b: &euchre_eval::runner::Contestant,
) {
    let pairs = opts.games / 2;
    let mut h2h = HeadToHead::default();
    for i in 0..pairs {
        h2h.record(run_pair(
            config,
            opts.seed.wrapping_add(i),
            &a.factory,
            &b.factory,
        ));
    }
    report(a, b, &h2h);
}

/// Plays pairs until the SPRT decides, or `--games` is exhausted.
fn run_sprt_mode(
    opts: &Options,
    config: GameConfig,
    a: &euchre_eval::runner::Contestant,
    b: &euchre_eval::runner::Contestant,
) {
    let sprt = Sprt::from_elo(opts.elo0, opts.elo1, opts.alpha, opts.beta);
    let (lower, upper) = sprt.bounds();
    println!(
        "SPRT H0: +{:.0} Elo  H1: +{:.0} Elo  (alpha={}, beta={}); LLR bounds [{:.3}, {:.3}]\n",
        opts.elo0, opts.elo1, opts.alpha, opts.beta, lower, upper
    );

    let max_pairs = opts.games / 2;
    let mut h2h = HeadToHead::default();
    let mut verdict = SprtVerdict::Continue;
    for i in 0..max_pairs {
        h2h.record(run_pair(
            config,
            opts.seed.wrapping_add(i),
            &a.factory,
            &b.factory,
        ));
        verdict = sprt.verdict(h2h.a_wins, h2h.b_wins);
        if verdict != SprtVerdict::Continue {
            break;
        }
    }

    match verdict {
        SprtVerdict::AcceptH1 => println!(
            "Result: H1 accepted — {} is stronger (LLR {:.3} >= {:.3}).",
            a.name,
            sprt.llr(h2h.a_wins, h2h.b_wins),
            upper
        ),
        SprtVerdict::AcceptH0 => println!(
            "Result: H0 accepted — {} is not meaningfully stronger (LLR {:.3} <= {:.3}).",
            a.name,
            sprt.llr(h2h.a_wins, h2h.b_wins),
            lower
        ),
        SprtVerdict::Continue => println!(
            "Result: undecided after {} games (LLR {:.3} within bounds); raise --games.",
            h2h.games,
            sprt.llr(h2h.a_wins, h2h.b_wins)
        ),
    }
    println!();
    report(a, b, &h2h);
}

/// Prints the shared summary block for a completed head-to-head.
fn report(
    a: &euchre_eval::runner::Contestant,
    b: &euchre_eval::runner::Contestant,
    h2h: &HeadToHead,
) {
    let (lo, hi) = wilson_interval(h2h.a_wins, h2h.games, Z_95);
    let test = mcnemar(h2h.a_better, h2h.b_better);
    println!("games:     {} ({} duplicate pairs)", h2h.games, h2h.pairs);
    println!(
        "record:    {} {}-{} {}",
        a.name, h2h.a_wins, h2h.b_wins, b.name
    );
    println!(
        "{} win rate: {:.1}%  (95% CI {:.1}–{:.1}%)",
        a.name,
        100.0 * h2h.a_win_rate(),
        100.0 * lo,
        100.0 * hi
    );
    println!(
        "McNemar:   {} better on {} deals, {} on {}; p = {:.4}",
        a.name, test.a_better, b.name, test.b_better, test.p_value
    );
}

/// Parses the command line, returning a human-readable error on failure.
fn parse_args() -> Result<Options, String> {
    let mut positional = Vec::new();
    let mut games = 2000_u64;
    let mut seed = 0_u64;
    let mut target_score = 10_u8;
    let mut sprt = false;
    let mut elo0 = 0.0_f64;
    let mut elo1 = 10.0_f64;
    let mut alpha = 0.05_f64;
    let mut beta = 0.05_f64;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = |flag: &str| {
            args.next()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match arg.as_str() {
            "--games" => games = next("--games")?.parse().map_err(|_| "invalid --games")?,
            "--seed" => seed = next("--seed")?.parse().map_err(|_| "invalid --seed")?,
            "--target-score" => {
                target_score = next("--target-score")?
                    .parse()
                    .map_err(|_| "invalid --target-score")?
            }
            "--sprt" => sprt = true,
            "--elo0" => elo0 = next("--elo0")?.parse().map_err(|_| "invalid --elo0")?,
            "--elo1" => elo1 = next("--elo1")?.parse().map_err(|_| "invalid --elo1")?,
            "--alpha" => alpha = next("--alpha")?.parse().map_err(|_| "invalid --alpha")?,
            "--beta" => beta = next("--beta")?.parse().map_err(|_| "invalid --beta")?,
            "-h" | "--help" => return Err("see usage below".into()),
            other if other.starts_with('-') => return Err(format!("unknown flag: {other}")),
            other => positional.push(other.to_string()),
        }
    }

    if positional.len() != 2 {
        return Err("expected exactly two agent names".into());
    }
    if games < 2 {
        return Err("--games must be at least 2".into());
    }
    if target_score == 0 {
        return Err("--target-score must be positive".into());
    }

    Ok(Options {
        agent_a: positional[0].clone(),
        agent_b: positional[1].clone(),
        games,
        seed,
        target_score,
        sprt,
        elo0,
        elo1,
        alpha,
        beta,
    })
}
