//! `euchre-play` — sit one human down at a terminal against three bots.
//!
//! This is the human-facing counterpart to the headless four-bot loop in the
//! crate docs: it wires a real keyboard and screen into one seat of a [`Driver`]
//! and fills the other three with [`AdvancedAgent`]s, then plays a full match to
//! the target score, narrating every bid, play, and trick so the human can
//! follow along.
//!
//! ```text
//! cargo run -p euchre-engine --bin euchre-play          # you sit South
//! cargo run -p euchre-engine --bin euchre-play -- east   # you sit East
//! ```
//!
//! The optional argument is the seat you take (`north`, `east`, `south`, or
//! `west`); it defaults to South, which by convention deals second. Everything
//! else — shuffling, scoring, turn order — is the engine's job.

use std::io::{self, BufReader, Write};
use std::process::ExitCode;

use euchre_agents::AdvancedAgent;
use euchre_engine::{Agent, Driver, GameConfig, Player, Seat, Verbosity};

fn main() -> ExitCode {
    let seat = match parse_seat(std::env::args().nth(1).as_deref()) {
        Ok(seat) => seat,
        Err(arg) => {
            eprintln!("Unknown seat {arg:?}. Choose one of: north, east, south, west.");
            return ExitCode::FAILURE;
        }
    };

    match play(seat) {
        Ok(()) => ExitCode::SUCCESS,
        // A broken pipe (e.g. the user closing the terminal) is a normal way to
        // quit, not an error worth a backtrace.
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Seats the human in `seat`, fills the rest with [`AdvancedAgent`]s, and runs a
/// match to completion against real stdin/stdout.
fn play(seat: Seat) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(
        out,
        "Euchre — you are {}. First to 10 points wins.",
        seat_name(seat)
    )?;
    writeln!(
        out,
        "Partners sit across; your partner is {}.\n",
        seat_name(partner(seat))
    )?;

    // Three independent bots for the seats the human is not occupying. They are
    // owned here and handed to the driver as `Player::Bot` slots in seat order.
    let mut bots = [
        AdvancedAgent::new(),
        AdvancedAgent::new(),
        AdvancedAgent::new(),
    ];
    let players = assemble(seat, &mut bots);

    let stdin = io::stdin();
    let input = BufReader::new(stdin.lock());

    let outcome = Driver::new(GameConfig::default(), players, Verbosity::Full, input, out).run()?;

    // The driver already narrated the winning line; nothing more to print, but
    // surface the result through the process for scripts that care.
    let _ = outcome;
    Ok(())
}

/// Builds the four [`Player`] slots in seat order (N, E, S, W), placing the
/// human in `human` and drawing bots from `bots` for the other three seats.
fn assemble<'a>(human: Seat, bots: &'a mut [AdvancedAgent; 3]) -> [Player<'a>; 4] {
    // Iterate the bots in order across the non-human seats so each appears once.
    let mut bots = bots.iter_mut();
    Seat::ALL.map(|seat| {
        if seat == human {
            Player::Human
        } else {
            Player::Bot(bots.next().expect("three bots for three seats") as &mut dyn Agent)
        }
    })
}

/// Parses the optional seat argument, defaulting to South. Returns the offending
/// argument on an unrecognized value.
fn parse_seat(arg: Option<&str>) -> Result<Seat, String> {
    match arg {
        None => Ok(Seat::South),
        Some(s) => match s.to_lowercase().as_str() {
            "north" | "n" => Ok(Seat::North),
            "east" | "e" => Ok(Seat::East),
            "south" | "s" => Ok(Seat::South),
            "west" | "w" => Ok(Seat::West),
            _ => Err(s.to_string()),
        },
    }
}

/// The seat across the table, i.e. the human's partner.
fn partner(seat: Seat) -> Seat {
    match seat {
        Seat::North => Seat::South,
        Seat::South => Seat::North,
        Seat::East => Seat::West,
        Seat::West => Seat::East,
    }
}

fn seat_name(seat: Seat) -> &'static str {
    match seat {
        Seat::North => "North",
        Seat::East => "East",
        Seat::South => "South",
        Seat::West => "West",
    }
}
