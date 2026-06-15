//! Round-robin tournament runner for Euchre agents.
//!
//! Plays every named agent against every other with duplicate dealing, fits
//! BayesElo-style Bradley-Terry ratings to the aggregate results, and prints a
//! leaderboard. CSV and JSON output make the ratings easy to track across runs.

use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use euchre_engine::GameConfig;
use euchre_eval::elo::{EloOptions, Rating, fit, leaderboard};
use euchre_eval::stats::{Z_95, wilson_interval};
use euchre_eval::tournament::{TournamentResults, matchups, run_round_robin};
use euchre_eval::{BUILTIN_AGENTS, builtin};

/// Run a round-robin tournament between Euchre agents and rank them by Elo.
///
/// Every agent plays every other with duplicate dealing (each deck played twice,
/// sides swapped, so deal luck cancels). Ratings are a batch Bradley-Terry fit
/// over all results with a small Bayesian prior, reported with standard errors.
#[derive(Debug, Parser)]
#[command(name = "euchre-tournament", version, about, long_about = None)]
struct Cli {
    /// Agents to enter (by name). Omit and pass --all to enter every built-in.
    agents: Vec<String>,

    /// Enter every built-in agent.
    #[arg(long)]
    all: bool,

    /// Matches per pairing (rounded down to an even number of duplicate pairs).
    #[arg(long, default_value_t = 500, value_parser = clap::value_parser!(u64).range(2..))]
    games: u64,

    /// Base deck seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Score a team must reach to win a match.
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u8).range(1..))]
    target_score: u8,

    /// Bayesian prior strength, in virtual even games versus a rating-0 anchor.
    #[arg(long, default_value_t = 2.0)]
    prior: f64,

    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Table)]
    format: Format,
}

/// How to render the standings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// Human-readable leaderboard plus a pairwise win-rate grid.
    Table,
    /// One row per agent, comma-separated, for spreadsheets and regression logs.
    Csv,
    /// A JSON array of rating records.
    Json,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let mut names: Vec<String> = Vec::new();
    if cli.all {
        names.extend(BUILTIN_AGENTS.iter().map(|s| s.to_string()));
    }
    names.extend(cli.agents.iter().cloned());
    names.dedup();

    if names.len() < 2 {
        eprintln!("need at least two agents (got {})", names.len());
        eprintln!("usage: euchre-tournament <agent> <agent> [more...]  (or --all)");
        eprintln!("known agents: {}", BUILTIN_AGENTS.join(", "));
        return ExitCode::FAILURE;
    }

    let mut contestants = Vec::with_capacity(names.len());
    for name in &names {
        match builtin(name) {
            Some(c) => contestants.push(c),
            None => {
                eprintln!("unknown agent: {name}");
                eprintln!("known agents: {}", BUILTIN_AGENTS.join(", "));
                return ExitCode::FAILURE;
            }
        }
    }

    let config = GameConfig {
        target_score: cli.target_score,
        ..GameConfig::default()
    };
    let pairs = cli.games / 2;

    if cli.format == Format::Table {
        eprintln!(
            "Round-robin: {} agents, {} pairings, {} matches each (to {} points)...",
            contestants.len(),
            matchups(contestants.len()).count(),
            2 * pairs,
            cli.target_score
        );
    }

    let results = run_round_robin(config, &contestants, pairs, cli.seed);
    let opts = EloOptions {
        prior_games: cli.prior,
        ..EloOptions::default()
    };
    let ratings = leaderboard(fit(&results.names, &results.wins_matrix(), &opts));

    match cli.format {
        Format::Table => print_table(&results, &ratings),
        Format::Csv => print_csv(&ratings),
        Format::Json => print_json(&ratings),
    }

    ExitCode::SUCCESS
}

/// Prints the leaderboard and a pairwise win-rate grid.
fn print_table(results: &TournamentResults, ratings: &[Rating]) {
    let name_w = ratings
        .iter()
        .map(|r| r.name.len())
        .chain(std::iter::once("agent".len()))
        .max()
        .unwrap_or(5);

    println!();
    println!(
        "  # {:<name_w$}  {:>10}  {:>9}  {:>6}  {:>6}  {:>7}",
        "agent", "elo", "± 95% CI", "wins", "losses", "win%",
    );
    for (rank, r) in ratings.iter().enumerate() {
        let (lo, hi) = wilson_interval(r.wins, r.games(), Z_95);
        println!(
            "{:>3} {:<name_w$}  {:>+10.0}  {:>9}  {:>6}  {:>6}  {:>6.1}%  ({:.0}–{:.0}% games)",
            rank + 1,
            r.name,
            r.elo,
            format!("± {:.0}", Z_95 * r.elo_stderr),
            r.wins,
            r.losses,
            100.0 * r.win_rate(),
            100.0 * lo,
            100.0 * hi,
        );
    }

    print_grid(results, ratings, name_w);
}

/// Prints a grid of each agent's match-win rate against each opponent, in
/// leaderboard order, so non-transitive matchups stand out.
fn print_grid(results: &TournamentResults, ratings: &[Rating], name_w: usize) {
    let wins = results.wins_matrix();
    // Map each name back to its row/column in the win matrix.
    let index = |name: &str| results.names.iter().position(|n| n == name).unwrap();
    let order: Vec<usize> = ratings.iter().map(|r| index(&r.name)).collect();

    println!("\npairwise match-win rate (row vs column):");
    print!("  {:<name_w$}", "");
    for r in ratings {
        print!("  {:>6}", short(&r.name));
    }
    println!();

    for (ri, &i) in order.iter().enumerate() {
        print!("  {:<name_w$}", ratings[ri].name);
        for &j in &order {
            if i == j {
                print!("  {:>6}", "—");
            } else {
                let played = wins[i][j] + wins[j][i];
                let rate = if played == 0 {
                    0.0
                } else {
                    100.0 * wins[i][j] as f64 / played as f64
                };
                print!("  {rate:>5.0}%");
            }
        }
        println!();
    }
}

/// Shortens a name to fit the grid's column width.
fn short(name: &str) -> String {
    if name.len() <= 6 {
        name.to_string()
    } else {
        name[..6].to_string()
    }
}

/// Prints the standings as CSV, leaderboard order.
fn print_csv(ratings: &[Rating]) {
    println!("rank,name,elo,elo_stderr,wins,losses,win_rate");
    for (rank, r) in ratings.iter().enumerate() {
        println!(
            "{},{},{:.1},{:.1},{},{},{:.4}",
            rank + 1,
            csv_escape(&r.name),
            r.elo,
            r.elo_stderr,
            r.wins,
            r.losses,
            r.win_rate(),
        );
    }
}

/// Escapes a field for CSV: quote and double inner quotes if it contains a
/// comma, quote, or newline. Agent names are simple today, but this keeps the
/// output well-formed regardless.
fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Prints the standings as a JSON array of rating records.
fn print_json(ratings: &[Rating]) {
    println!("[");
    for (k, r) in ratings.iter().enumerate() {
        let comma = if k + 1 < ratings.len() { "," } else { "" };
        println!(
            "  {{\"rank\": {}, \"name\": {}, \"elo\": {:.1}, \"elo_stderr\": {:.1}, \
             \"wins\": {}, \"losses\": {}, \"win_rate\": {:.4}}}{}",
            k + 1,
            json_string(&r.name),
            r.elo,
            r.elo_stderr,
            r.wins,
            r.losses,
            r.win_rate(),
            comma,
        );
    }
    println!("]");
}

/// Renders a string as a JSON string literal, escaping the characters JSON
/// requires (quote, backslash, control characters).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
