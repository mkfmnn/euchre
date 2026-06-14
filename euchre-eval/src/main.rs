//! Command-line head-to-head runner for Euchre agents.
//!
//! In fixed-games mode it reports the first agent's win rate with a Wilson 95%
//! interval and McNemar's paired p-value. With `--sprt` it plays duplicate pairs
//! until the sequential test accepts H0 or H1 (or `--games` is exhausted).

use std::process::ExitCode;

use clap::Parser;
use euchre_engine::GameConfig;
use euchre_eval::runner::{Contestant, HeadToHead, run_pair};
use euchre_eval::stats::{Sprt, SprtVerdict, Z_95, mcnemar, wilson_interval};
use euchre_eval::{BUILTIN_AGENTS, builtin};

/// Play two Euchre agents head to head and report who wins more matches.
///
/// Matches are run with duplicate dealing: every deck is played twice with the
/// agents on opposite sides, so deal luck cancels.
#[derive(Debug, Parser)]
#[command(name = "euchre-eval", version, about, long_about = None)]
struct Cli {
    /// First agent (reported as "A").
    agent_a: String,
    /// Second agent (reported as "B").
    agent_b: String,

    /// Number of matches to play (rounded down to an even number of pairs).
    #[arg(long, default_value_t = 2000, value_parser = clap::value_parser!(u64).range(2..))]
    games: u64,

    /// Base deck seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Score a team must reach to win a match.
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u8).range(1..))]
    target_score: u8,

    /// Stop early once the result is decided (sequential probability ratio test).
    #[arg(long)]
    sprt: bool,

    /// H0 Elo gain for the SPRT (the "not worth it" hypothesis).
    #[arg(long, default_value_t = 0.0)]
    elo0: f64,

    /// H1 Elo gain for the SPRT (the "real improvement" hypothesis).
    #[arg(long, default_value_t = 10.0)]
    elo1: f64,

    /// SPRT type-I error rate (false-positive: accept H1 when H0 holds).
    #[arg(long, default_value_t = 0.05)]
    alpha: f64,

    /// SPRT type-II error rate (false-negative: accept H0 when H1 holds).
    #[arg(long, default_value_t = 0.05)]
    beta: f64,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let Some(a) = builtin(&cli.agent_a) else {
        eprintln!("unknown agent: {}", cli.agent_a);
        eprintln!("known agents: {}", BUILTIN_AGENTS.join(", "));
        return ExitCode::FAILURE;
    };
    let Some(b) = builtin(&cli.agent_b) else {
        eprintln!("unknown agent: {}", cli.agent_b);
        eprintln!("known agents: {}", BUILTIN_AGENTS.join(", "));
        return ExitCode::FAILURE;
    };

    let config = GameConfig {
        target_score: cli.target_score,
        ..GameConfig::default()
    };

    println!(
        "{} (A) vs {} (B) — to {} points, duplicate dealing\n",
        a.name, b.name, cli.target_score
    );

    if cli.sprt {
        run_sprt_mode(&cli, config, &a, &b);
    } else {
        run_fixed_mode(&cli, config, &a, &b);
    }
    ExitCode::SUCCESS
}

/// Plays a fixed number of pairs and prints win rate, Wilson interval and McNemar.
fn run_fixed_mode(cli: &Cli, config: GameConfig, a: &Contestant, b: &Contestant) {
    let pairs = cli.games / 2;
    let mut h2h = HeadToHead::default();
    for i in 0..pairs {
        h2h.record(run_pair(
            config,
            cli.seed.wrapping_add(i),
            &a.factory,
            &b.factory,
        ));
    }
    report(a, b, &h2h);
}

/// Plays pairs until the SPRT decides, or `--games` is exhausted.
fn run_sprt_mode(cli: &Cli, config: GameConfig, a: &Contestant, b: &Contestant) {
    let sprt = Sprt::from_elo(cli.elo0, cli.elo1, cli.alpha, cli.beta);
    let (lower, upper) = sprt.bounds();
    println!(
        "SPRT H0: +{:.0} Elo  H1: +{:.0} Elo  (alpha={}, beta={}); LLR bounds [{:.3}, {:.3}]\n",
        cli.elo0, cli.elo1, cli.alpha, cli.beta, lower, upper
    );

    let max_pairs = cli.games / 2;
    let mut h2h = HeadToHead::default();
    let mut verdict = SprtVerdict::Continue;
    for i in 0..max_pairs {
        h2h.record(run_pair(
            config,
            cli.seed.wrapping_add(i),
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
fn report(a: &Contestant, b: &Contestant, h2h: &HeadToHead) {
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
