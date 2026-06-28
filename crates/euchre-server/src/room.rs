//! The room actor: one task that owns a table — its lobby and the [`Game`] it
//! drives over the wire.
//!
//! A single task owning the game keeps the engine's "decisions are sequential,
//! never concurrent" guarantee for free. Connection tasks talk to the room only
//! by sending [`RoomMsg`]s down an mpsc channel; the room talks back to each
//! connection by pushing [`ServerMsg`]s into that connection's own channel.
//!
//! A room has two phases that alternate:
//!
//! * **Lobby** — connections come and go and arrange the four seats with
//!   [`ClientMsg::Seat`] requests. When all four seats stay occupied for
//!   [`AUTOSTART`], a match begins.
//! * **Match** — the async, networked analogue of the terminal
//!   [`Driver`](euchre_engine::Driver): ask the core
//!   [what is needed](Game::next_action), route it to a bot (call the agent
//!   directly) or a human (send `Awaiting`, await their reply), then
//!   [apply](Game::apply) and broadcast what happened. When the match ends the
//!   room returns to the lobby.
//!
//! ## Identity
//!
//! A connection is identified by a `conn_id`, decoupled from any seat. The
//! engine names seats *relative to the dealer* ([`Seat`]), which rotates each
//! hand; the room pins each seat to a fixed table position ([`Player`]: `0` =
//! North … `3` = West) and translates with [`Game::player_at`] /
//! [`Game::seat_of`] at the boundary. The wire protocol speaks only fixed
//! positions.

use std::collections::HashMap;
use std::time::Duration;

use euchre_agents::HeuristicAgent;
use euchre_engine::{Action, Decision, Game, GameConfig};
use euchre_interface::{Agent, GameView, Seat};
use rand::rngs::ChaCha12Rng;
use rand::{RngExt, SeedableRng};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

use crate::Registry;
use crate::protocol::{ClientMsg, Player, SeatInfo, SeatRequest, ServerMsg};
use crate::view::{decision_from, hint_for, public_action};

/// How long the room waits for a human's move before substituting a bot one, so
/// a slow or vanished player cannot wedge the table.
const TURN_TIMEOUT: Duration = Duration::from_secs(120);

/// How long all four seats must stay occupied before a match starts.
const AUTOSTART: Duration = Duration::from_secs(5);

/// A message from a connection task to the room.
pub enum RoomMsg {
    /// A client joined the table; the room registers it and acks once it has
    /// (and has sent the first [`ServerMsg::TableState`]).
    Join {
        conn_id: u64,
        name: String,
        out: mpsc::UnboundedSender<ServerMsg>,
        ack: oneshot::Sender<()>,
    },
    /// A client sent a message, tagged with its connection id. The room decides
    /// what it means given the current phase.
    Msg { conn_id: u64, msg: ClientMsg },
    /// A client's socket closed.
    Disconnect { conn_id: u64 },
}

/// A connected client: its display name and the channel to its socket.
struct Conn {
    name: String,
    out: mpsc::UnboundedSender<ServerMsg>,
}

/// Who occupies a fixed table position.
enum SeatSlot {
    /// Nobody.
    Empty,
    /// A server-side bot.
    Bot {
        name: String,
        agent: Box<dyn Agent + Send>,
    },
    /// A connected human, referenced by connection id (the display name lives in
    /// [`Room::connections`]).
    Human { conn_id: u64 },
}

/// The room: owns the lobby, the (in-progress) game, the four seats, and the
/// shuffler.
pub struct Room {
    config: GameConfig,
    /// This table's short code, also its key in the [`Registry`].
    code: String,
    /// The shared table registry, so the room can remove itself when it ends.
    registry: Registry,
    /// The match in progress, or `None` while in the lobby.
    game: Option<Game>,
    rng: ChaCha12Rng,
    /// Everyone connected to the table, seated or not, by connection id.
    connections: HashMap<u64, Conn>,
    /// The four seats, indexed by fixed table position ([`Player`]).
    seats: [SeatSlot; 4],
    /// Used to compute a legal move when a human times out or disconnects.
    fallback: HeuristicAgent,
    rx: mpsc::UnboundedReceiver<RoomMsg>,
}

/// The outcome of waiting for a human's action.
enum HumanWait {
    /// The human acted.
    Got(ClientMsg),
    /// Timed out or the human vanished — substitute a bot move.
    Fallback,
    /// No connections remain — abandon the match.
    Abandon,
}

impl Room {
    /// Creates an empty table: no connections, all seats open, no game yet.
    pub fn new(
        config: GameConfig,
        code: String,
        registry: Registry,
        rx: mpsc::UnboundedReceiver<RoomMsg>,
    ) -> Self {
        Room {
            config,
            code,
            registry,
            game: None,
            rng: ChaCha12Rng::from_rng(&mut rand::rng()),
            connections: HashMap::new(),
            seats: [
                SeatSlot::Empty,
                SeatSlot::Empty,
                SeatSlot::Empty,
                SeatSlot::Empty,
            ],
            fallback: HeuristicAgent::new(),
            rx,
        }
    }

    /// Runs the table forever: arrange seats in the lobby, play a match, return
    /// to the lobby, until the last connection leaves.
    pub async fn run(mut self) {
        loop {
            if !self.lobby().await {
                break;
            }
            self.start_game();
            self.play_match().await;
            self.game = None;
            if self.connections.is_empty() {
                break;
            }
        }
        self.registry.lock().unwrap().remove(&self.code);
        tracing::info!(code = %self.code, "table closed");
    }

    // --- Lobby ---------------------------------------------------------------

    /// Runs the lobby until all four seats stay occupied for [`AUTOSTART`],
    /// returning `true` to start a match, or `false` if every connection left.
    async fn lobby(&mut self) -> bool {
        // Reflect the current seating (e.g. when returning from a finished match).
        self.broadcast_table_state();
        let mut deadline: Option<Instant> = self.countdown_deadline(None);
        loop {
            tokio::select! {
                _ = sleep_or_pending(deadline), if deadline.is_some() => return true,
                msg = self.rx.recv() => {
                    match msg {
                        None => return false,
                        Some(RoomMsg::Join { conn_id, name, out, ack }) => {
                            self.connections.insert(conn_id, Conn { name, out });
                            let _ = ack.send(());
                            self.broadcast_table_state();
                        }
                        Some(RoomMsg::Disconnect { conn_id }) => {
                            self.connections.remove(&conn_id);
                            self.vacate_human(conn_id);
                            if self.connections.is_empty() {
                                return false;
                            }
                            self.broadcast_table_state();
                        }
                        Some(RoomMsg::Msg { conn_id, msg }) => {
                            if let ClientMsg::Seat { seat, player } = msg {
                                self.handle_seat(conn_id, seat, player);
                                self.broadcast_table_state();
                            }
                            // Game actions arriving in the lobby are ignored.
                        }
                    }
                    deadline = self.countdown_deadline(deadline);
                }
            }
        }
    }

    /// The auto-start deadline: keep any running one while all four seats stay
    /// occupied, start a fresh one when they first all fill, and clear it as
    /// soon as a seat opens.
    fn countdown_deadline(&self, current: Option<Instant>) -> Option<Instant> {
        if self.all_seated() {
            current.or_else(|| Some(Instant::now() + AUTOSTART))
        } else {
            None
        }
    }

    fn all_seated(&self) -> bool {
        self.seats.iter().all(|s| !matches!(s, SeatSlot::Empty))
    }

    /// Applies a seating request, sending the requester an error if it is not
    /// allowed. The caller broadcasts the resulting table state.
    fn handle_seat(&mut self, conn_id: u64, seat: Player, req: SeatRequest) {
        let p = seat as usize;
        if p >= 4 {
            return;
        }
        match req {
            SeatRequest::Me => {
                if !matches!(self.seats[p], SeatSlot::Empty) {
                    self.error_to(conn_id, "that seat is taken");
                    return;
                }
                self.vacate_human(conn_id); // a human holds at most one seat
                self.seats[p] = SeatSlot::Human { conn_id };
            }
            SeatRequest::Bot => {
                if !matches!(self.seats[p], SeatSlot::Empty) {
                    self.error_to(conn_id, "that seat is taken");
                    return;
                }
                self.seats[p] = bot_for(p);
            }
            SeatRequest::Empty => match self.seats[p] {
                SeatSlot::Bot { .. } => self.seats[p] = SeatSlot::Empty,
                SeatSlot::Human { conn_id: c } if c == conn_id => self.seats[p] = SeatSlot::Empty,
                SeatSlot::Human { .. } => {
                    self.error_to(conn_id, "you can only vacate your own seat")
                }
                SeatSlot::Empty => {}
            },
        }
    }

    /// Empties whatever seat `conn_id` holds, if any.
    fn vacate_human(&mut self, conn_id: u64) {
        for slot in &mut self.seats {
            if matches!(slot, SeatSlot::Human { conn_id: c } if *c == conn_id) {
                *slot = SeatSlot::Empty;
            }
        }
    }

    // --- Match ---------------------------------------------------------------

    /// Builds a fresh game with a random first dealer and announces it.
    fn start_game(&mut self) {
        let first_dealer: usize = self.rng.random_range(0..4);
        let mut config = self.config;
        config.first_dealer = first_dealer;
        self.game = Some(Game::new(config, euchre_engine::deal(&mut self.rng)));
        self.broadcast(ServerMsg::StartGame {
            first_dealer: first_dealer as Player,
        });
    }

    /// Plays the match to its end (or until everyone leaves), broadcasting the
    /// event stream. Returns when the match is over.
    async fn play_match(&mut self) {
        self.broadcast_deal();
        loop {
            let action = self.game().next_action();
            match action {
                Action::HandComplete { .. } => {
                    self.notify_hand_end();
                    self.broadcast_hand_complete();
                    if self.game().is_over() {
                        let winner = self.game().winner().expect("decided match has a winner");
                        self.broadcast(ServerMsg::GameOver {
                            winner: winner as u8,
                            scores: self.game().scores(),
                        });
                        return; // back to the lobby
                    }
                    let deck = euchre_engine::deal(&mut self.rng);
                    self.game_mut()
                        .start_next_hand(deck)
                        .expect("ready for next hand");
                    self.broadcast_deal();
                }
                action => {
                    let seat = action_seat(&action);
                    let player = self.game().player_at(seat);
                    self.broadcast_awaiting(&action, seat);
                    let decision = match self.decide(&action, seat).await {
                        Some(decision) => decision,
                        None => return, // everyone left; abandon the match
                    };
                    let tricks_before = self.game().completed_tricks().len();
                    match self.game_mut().apply(decision) {
                        Ok(()) => {
                            self.broadcast(ServerMsg::Update {
                                player: player as u8,
                                action: public_action(&action, &decision),
                            });
                            if self.game().completed_tricks().len() > tricks_before {
                                let (_, winner) = *self
                                    .game()
                                    .completed_tricks()
                                    .last()
                                    .expect("a trick just completed");
                                self.broadcast(ServerMsg::TrickWon {
                                    player: self.game().player_at(winner) as u8,
                                });
                            }
                        }
                        Err(e) => {
                            // A bot should never produce an illegal move; a human
                            // might (e.g. a card that fails to follow suit). Tell
                            // them and re-ask by looping with the same action.
                            self.send_to(
                                player,
                                ServerMsg::Error {
                                    message: e.to_string(),
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    /// Obtains the decision for the seat on turn, from a bot directly or a human
    /// over the wire (falling back to a bot move on timeout/disconnect).
    /// `None` means every connection has left and the match should be abandoned.
    async fn decide(&mut self, action: &Action, seat: Seat) -> Option<Decision> {
        let player = self.game().player_at(seat);
        if !self.is_human(player) {
            return Some(self.bot_decision(action, seat));
        }
        loop {
            match self.await_human_action(player).await {
                HumanWait::Got(msg) => match decision_from(&msg, action) {
                    Ok(decision) => return Some(decision),
                    Err(message) => {
                        self.send_to(player, ServerMsg::Error { message });
                        self.broadcast_awaiting(action, seat);
                    }
                },
                HumanWait::Fallback => return Some(self.fallback_decision(action, seat)),
                HumanWait::Abandon => return None,
            }
        }
    }

    /// Waits for the action of the human in `player`'s seat, handling
    /// joins/disconnects meanwhile.
    async fn await_human_action(&mut self, player: usize) -> HumanWait {
        let active_conn = match self.seats[player] {
            SeatSlot::Human { conn_id } => conn_id,
            _ => return HumanWait::Fallback, // seat is a bot now
        };
        let deadline = Instant::now() + TURN_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match tokio::time::timeout(remaining, self.rx.recv()).await {
                Err(_) => return HumanWait::Fallback,  // timed out
                Ok(None) => return HumanWait::Abandon, // channel closed
                Ok(Some(msg)) => match msg {
                    RoomMsg::Msg { conn_id, msg } if conn_id == active_conn => {
                        if matches!(msg, ClientMsg::Seat { .. }) {
                            self.error_to(active_conn, "the game is already in progress");
                        } else {
                            return HumanWait::Got(msg);
                        }
                    }
                    RoomMsg::Msg { .. } => {} // out of turn or a spectator; ignore
                    RoomMsg::Join {
                        conn_id,
                        name,
                        out,
                        ack,
                    } => {
                        // A latecomer to a table mid-match: register them and show
                        // the (full) table; they wait for the next match.
                        let _ = out.send(ServerMsg::TableState {
                            table: self.code.clone(),
                            your_seat: None,
                            seats: self.seat_infos(),
                        });
                        self.connections.insert(conn_id, Conn { name, out });
                        let _ = ack.send(());
                    }
                    RoomMsg::Disconnect { conn_id } => {
                        self.handle_disconnect_match(conn_id);
                        if self.connections.is_empty() {
                            return HumanWait::Abandon;
                        }
                        if conn_id == active_conn {
                            return HumanWait::Fallback;
                        }
                    }
                },
            }
        }
    }

    /// A connection dropped mid-match: drop it and, if it held a seat, hand that
    /// seat to a bot so the match can continue.
    fn handle_disconnect_match(&mut self, conn_id: u64) {
        self.connections.remove(&conn_id);
        for player in 0..4 {
            if matches!(self.seats[player], SeatSlot::Human { conn_id: c } if c == conn_id) {
                self.seats[player] = bot_for(player);
            }
        }
    }

    fn bot_decision(&mut self, action: &Action, seat: Seat) -> Decision {
        let game = self.game.as_ref().expect("match in progress");
        let player = game.player_at(seat);
        let view = game.view(seat);
        match &mut self.seats[player] {
            SeatSlot::Bot { agent, .. } => agent_decide(agent.as_mut(), action, &view),
            _ => unreachable!("bot_decision on a non-bot seat"),
        }
    }

    fn fallback_decision(&mut self, action: &Action, seat: Seat) -> Decision {
        let view = self.game.as_ref().expect("match in progress").view(seat);
        agent_decide(&mut self.fallback, action, &view)
    }

    /// Lets seated bots observe how the hand ended, from their own seat's point of
    /// view (stateful agents can learn).
    fn notify_hand_end(&mut self) {
        let game = self.game.as_ref().expect("match in progress");
        for player in 0..4 {
            let seat = game.seat_of(player);
            let view = game.view(seat);
            let result = game.hand_result(seat);
            if let SeatSlot::Bot { agent, .. } = &mut self.seats[player] {
                agent.observe_hand_end(&view, &result);
            }
        }
    }

    fn is_human(&self, player: usize) -> bool {
        matches!(self.seats[player], SeatSlot::Human { .. })
    }

    fn game(&self) -> &Game {
        self.game.as_ref().expect("match in progress")
    }

    fn game_mut(&mut self) -> &mut Game {
        self.game.as_mut().expect("match in progress")
    }

    // --- Sending -------------------------------------------------------------

    /// The connection seated at `player`, if it is a (still-connected) human.
    fn seat_conn(&self, player: usize) -> Option<&Conn> {
        match &self.seats[player] {
            SeatSlot::Human { conn_id } => self.connections.get(conn_id),
            _ => None,
        }
    }

    fn seat_infos(&self) -> [SeatInfo; 4] {
        std::array::from_fn(|p| match &self.seats[p] {
            SeatSlot::Empty => SeatInfo::Empty,
            SeatSlot::Bot { name, .. } => SeatInfo::Bot { name: name.clone() },
            SeatSlot::Human { conn_id } => SeatInfo::Human {
                name: self
                    .connections
                    .get(conn_id)
                    .map(|c| c.name.clone())
                    .unwrap_or_default(),
            },
        })
    }

    /// The seat held by `conn_id`, if any.
    fn seat_of_conn(&self, conn_id: u64) -> Option<Player> {
        (0..4).find_map(|p| match self.seats[p] {
            SeatSlot::Human { conn_id: c } if c == conn_id => Some(p as Player),
            _ => None,
        })
    }

    /// Sends every connection the lobby snapshot, each with its own `your_seat`.
    fn broadcast_table_state(&self) {
        let seats = self.seat_infos();
        for (&conn_id, conn) in &self.connections {
            let _ = conn.out.send(ServerMsg::TableState {
                table: self.code.clone(),
                your_seat: self.seat_of_conn(conn_id),
                seats: seats.clone(),
            });
        }
    }

    fn broadcast_deal(&self) {
        let game = self.game();
        let dealer = game.dealer() as u8;
        let up_card = game.up_card();
        for player in 0..4 {
            if let Some(conn) = self.seat_conn(player) {
                let seat = game.seat_of(player);
                let _ = conn.out.send(ServerMsg::Deal {
                    dealer,
                    hand: game.hand(seat).to_vec(),
                    up_card,
                });
            }
        }
    }

    /// Sends each seated human the just-completed hand's result, told from their
    /// own seat's point of view.
    fn broadcast_hand_complete(&self) {
        let game = self.game();
        for player in 0..4 {
            if let Some(conn) = self.seat_conn(player) {
                let seat = game.seat_of(player);
                let _ = conn.out.send(ServerMsg::HandComplete {
                    result: game.hand_result(seat),
                });
            }
        }
    }

    /// Tells everyone whose turn it is; the active human also gets its legal
    /// plays (a card's legality can reveal a void, so only that seat sees them).
    fn broadcast_awaiting(&self, action: &Action, active: Seat) {
        let game = self.game();
        let hint = hint_for(action, game);
        let legal = match action {
            Action::Play { legal, .. } => Some(legal.clone()),
            _ => None,
        };
        let active_player = game.player_at(active) as u8;
        for player in 0..4 {
            if let Some(conn) = self.seat_conn(player) {
                let legal = if player as u8 == active_player {
                    legal.clone()
                } else {
                    None
                };
                let _ = conn.out.send(ServerMsg::Awaiting {
                    player: active_player,
                    hint: hint.clone(),
                    legal,
                });
            }
        }
    }

    fn broadcast(&self, msg: ServerMsg) {
        for player in 0..4 {
            if let Some(conn) = self.seat_conn(player) {
                let _ = conn.out.send(msg.clone());
            }
        }
    }

    fn send_to(&self, player: usize, msg: ServerMsg) {
        if let Some(conn) = self.seat_conn(player) {
            let _ = conn.out.send(msg);
        }
    }

    fn error_to(&self, conn_id: u64, message: &str) {
        if let Some(conn) = self.connections.get(&conn_id) {
            let _ = conn.out.send(ServerMsg::Error {
                message: message.to_string(),
            });
        }
    }
}

/// A future that sleeps until `deadline`, or never resolves if there is none.
async fn sleep_or_pending(deadline: Option<Instant>) {
    match deadline {
        Some(d) => tokio::time::sleep_until(d).await,
        None => std::future::pending().await,
    }
}

/// Asks an agent for its decision to the pending `action`.
fn agent_decide(agent: &mut (dyn Agent + Send), action: &Action, view: &GameView<'_>) -> Decision {
    match action {
        Action::BidUpcard { .. } => Decision::Upcard(agent.bid_upcard(view)),
        Action::BidCall { .. } => Decision::Call(agent.bid_call(view)),
        Action::Discard { .. } => Decision::Discard(agent.discard(view)),
        Action::Play { legal, .. } => Decision::Play(agent.play_card(view, legal)),
        Action::HandComplete { .. } => unreachable!("HandComplete asks no agent"),
    }
}

/// The acting seat of an action (never called for `HandComplete`).
fn action_seat(action: &Action) -> Seat {
    match action {
        Action::BidUpcard { seat, .. }
        | Action::BidCall { seat, .. }
        | Action::Discard { seat, .. }
        | Action::Play { seat, .. } => *seat,
        Action::HandComplete { .. } => unreachable!("HandComplete has no acting seat"),
    }
}

fn bot_for(player: usize) -> SeatSlot {
    SeatSlot::Bot {
        name: format!("Bot {}", player_name(player)),
        agent: Box::new(HeuristicAgent::new()),
    }
}

/// The conventional name of a fixed table position (`0` = North … `3` = West).
fn player_name(player: usize) -> &'static str {
    ["North", "East", "South", "West"][player]
}
