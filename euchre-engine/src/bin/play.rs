//! An interactive Euchre game for the terminal.
//!
//! Seats a human in the South chair against three [`HeuristicAgent`] bots and
//! plays a full match to 10 points, narrating the action as it goes.
//!
//! Run it with:
//!
//! ```text
//! cargo run -p euchre-engine
//! ```
//!
//! An optional integer argument seeds the deck for a reproducible game:
//!
//! ```text
//! cargo run -p euchre-engine -- 42
//! ```

use std::io::{self, Write};

use euchre_engine::agents::{HeuristicAgent, HumanAgent};
use euchre_engine::engine::{Engine, EngineConfig, HandOutcome};
use euchre_interface::{Agent, Team};

fn main() {
    let seed = std::env::args().nth(1).and_then(|a| a.parse::<u64>().ok());

    println!("╔══════════════════════════════════════════╗");
    println!("║              E U C H R E                 ║");
    println!("╚══════════════════════════════════════════╝");
    println!();
    println!("You are South. Your partner is North; East and West are opponents.");
    println!("First team to 10 points wins. Good luck!\n");

    // South is the human; the rest are heuristic bots.
    let agents: [Box<dyn Agent>; 4] = [
        Box::new(HeuristicAgent::new()),  // North
        Box::new(HeuristicAgent::new()),  // East
        Box::new(HumanAgent::terminal()), // South
        Box::new(HeuristicAgent::new()),  // West
    ];

    let config = EngineConfig {
        seed,
        ..Default::default()
    };
    let mut engine = Engine::new(agents, config);

    let mut hand_no = 0;
    while !engine.is_over() {
        hand_no += 1;
        let dealer = engine.dealer();
        println!("\n══════════ Hand {hand_no} ══════════  (dealer: {dealer:?})");

        match engine.play_hand() {
            Ok(HandOutcome::Played(_)) => {}
            Ok(HandOutcome::PassedOut) => {
                println!("Everyone passed — the hand is thrown in.");
            }
            Err(e) => {
                eprintln!("Engine error: {e}");
                break;
            }
        }

        let scores = engine.scores();
        println!(
            "Score:  N/S {}  –  E/W {}",
            scores.north_south, scores.east_west
        );
        let _ = io::stdout().flush();
    }

    let scores = engine.scores();
    let winner = if scores.north_south >= config.target_score {
        Team::NorthSouth
    } else {
        Team::EastWest
    };
    println!("\n════════════════════════════════════════════");
    match winner {
        Team::NorthSouth => println!(
            "Your team (N/S) wins {}–{}! 🎉",
            scores.north_south, scores.east_west
        ),
        Team::EastWest => println!(
            "The opponents (E/W) win {}–{}. Better luck next time.",
            scores.east_west, scores.north_south
        ),
    }
}
