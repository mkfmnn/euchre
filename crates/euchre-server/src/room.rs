//! The room actor: one task that owns a [`Game`] and drives a match over the
//! wire.
//!
//! A single task owning the game keeps the engine's "decisions are sequential,
//! never concurrent" guarantee for free. Connection tasks talk to the room only
//! by sending [`RoomMsg`]s down an mpsc channel; the room talks back to each
//! human by pushing [`ServerMsg`]s into that connection's own channel.
//!
//! The loop is the async, networked analogue of the terminal
//! [`Driver`](euchre_engine::Driver): ask the core
//! [what is needed](Game::next_action), route it to a bot (call the agent
//! directly) or a human (send `Awaiting`, await their reply), then
//! [apply](Game::apply) and broadcast what happened.
//!
//! ## Identity
//!
//! The engine names seats *relative to the dealer* ([`Seat`]), which rotates each
//! hand. The room instead pins each occupant to a fixed table position
//! ([`Player`]: `0` = North … `3` = West) that never changes, and translates to
//! and from the engine's dealer-relative seat with [`Game::player_at`] /
//! [`Game::seat_of`] at the boundary. The wire protocol speaks only fixed
//! positions.

use std::time::Duration;

use euchre_agents::HeuristicAgent;
use euchre_engine::{Action, Decision, Game, GameConfig};
use euchre_interface::{Agent, GameView, Seat};
use rand::SeedableRng;
use rand::rngs::ChaCha12Rng;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

use crate::protocol::{ClientMsg, Player, SeatedPlayer, ServerMsg};
use crate::view::{decision_from, hint_for, public_action, snapshot};

/// How long the room waits for a human's move before substituting a bot one, so
/// a slow or vanished player cannot wedge the table.
const TURN_TIMEOUT: Duration = Duration::from_secs(120);

/// A message from a connection task to the room.
pub enum RoomMsg {
    /// A client introduced itself and wants a seat. The room replies with the
    /// assigned table position over `ack`, or `None` if the table is full.
    Hello {
        name: String,
        seat: Option<Player>,
        out: mpsc::UnboundedSender<ServerMsg>,
        ack: oneshot::Sender<Option<Player>>,
    },
    /// A seated client sent an action, tagged with its fixed table position.
    Action { player: Player, msg: ClientMsg },
    /// A client's socket closed; free its position.
    Disconnect { player: Player },
}

/// Who occupies a table position.
enum Occupant {
    /// Not yet assigned (only before the first human joins).
    Open,
    /// A connected human; `out` pushes messages to their socket.
    Human {
        name: String,
        out: mpsc::UnboundedSender<ServerMsg>,
    },
    /// A server-side bot.
    Bot {
        name: String,
        agent: Box<dyn Agent + Send>,
    },
}

/// The room: owns the game, the four occupants, and the shuffler.
pub struct Room {
    config: GameConfig,
    game: Game,
    rng: ChaCha12Rng,
    /// The four occupants, indexed by fixed table position ([`Player`]).
    occupants: [Occupant; 4],
    /// Used to compute a legal move when a human times out or disconnects.
    fallback: HeuristicAgent,
    rx: mpsc::UnboundedReceiver<RoomMsg>,
}

impl Room {
    /// Creates a room with all seats open and the first hand dealt.
    pub fn new(config: GameConfig, rx: mpsc::UnboundedReceiver<RoomMsg>) -> Self {
        let mut rng = ChaCha12Rng::from_rng(&mut rand::rng());
        let game = Game::new(config, euchre_engine::deal(&mut rng));
        Room {
            config,
            game,
            rng,
            occupants: [
                Occupant::Open,
                Occupant::Open,
                Occupant::Open,
                Occupant::Open,
            ],
            fallback: HeuristicAgent::new(),
            rx,
        }
    }

    /// Runs the room forever: wait for the first human, then play matches
    /// back-to-back, starting a fresh one after each `GameOver`.
    pub async fn run(mut self) {
        if !self.wait_for_first_human().await {
            return; // channel closed before anyone joined
        }
        self.broadcast_deal();

        loop {
            let action = self.game.next_action();
            match action {
                Action::HandComplete { .. } => {
                    self.notify_hand_end();
                    self.broadcast_hand_complete();
                    if self.game.is_over() {
                        let winner = self.game.winner().expect("decided match has a winner");
                        self.broadcast(ServerMsg::GameOver {
                            winner: winner as u8,
                            scores: self.game.scores(),
                        });
                        // Start a fresh match; the seated players carry over.
                        self.game = Game::new(self.config, euchre_engine::deal(&mut self.rng));
                    } else {
                        let deck = euchre_engine::deal(&mut self.rng);
                        self.game
                            .start_next_hand(deck)
                            .expect("ready for next hand");
                    }
                    self.broadcast_deal();
                }
                action => {
                    let seat = action_seat(&action);
                    let player = self.game.player_at(seat);
                    self.broadcast_awaiting(&action, seat);
                    let decision = self.decide(&action, seat).await;
                    let tricks_before = self.game.completed_tricks().len();
                    match self.game.apply(decision) {
                        Ok(()) => {
                            self.broadcast(ServerMsg::Update {
                                player: player as u8,
                                action: public_action(&action, &decision),
                            });
                            if self.game.completed_tricks().len() > tricks_before {
                                let (_, winner) = *self
                                    .game
                                    .completed_tricks()
                                    .last()
                                    .expect("a trick just completed");
                                self.broadcast(ServerMsg::TrickWon {
                                    player: self.game.player_at(winner) as u8,
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

    /// Blocks until a human joins (seating them and filling the rest with bots),
    /// returning `false` if the channel closes first.
    async fn wait_for_first_human(&mut self) -> bool {
        loop {
            match self.rx.recv().await {
                Some(RoomMsg::Hello {
                    name,
                    seat,
                    out,
                    ack,
                }) => {
                    self.handle_hello(name, seat, out, ack);
                    return true;
                }
                Some(_) => {} // no game in progress yet; ignore stray actions
                None => return false,
            }
        }
    }

    /// Obtains the decision for the seat on turn, from a bot directly or a human
    /// over the wire (falling back to a bot move on timeout/disconnect).
    async fn decide(&mut self, action: &Action, seat: Seat) -> Decision {
        let player = self.game.player_at(seat);
        if !self.is_human(player) {
            return self.bot_decision(action, seat);
        }
        loop {
            match self.await_human_action(player).await {
                Some(msg) => match decision_from(&msg, action) {
                    Ok(decision) => return decision,
                    Err(message) => {
                        self.send_to(player, ServerMsg::Error { message });
                        self.broadcast_awaiting(action, seat);
                    }
                },
                None => return self.fallback_decision(action, seat),
            }
        }
    }

    /// Waits for `player`'s action, handling joins/disconnects meanwhile. Returns
    /// `None` on timeout, on this player disconnecting, or on channel close — all
    /// of which mean "substitute a bot move".
    async fn await_human_action(&mut self, player: usize) -> Option<ClientMsg> {
        let deadline = Instant::now() + TURN_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match tokio::time::timeout(remaining, self.rx.recv()).await {
                Err(_) => return None,   // timed out
                Ok(None) => return None, // channel closed
                Ok(Some(msg)) => match msg {
                    RoomMsg::Action { player: p, msg } if p as usize == player => return Some(msg),
                    RoomMsg::Action { .. } => {} // out of turn; ignore
                    RoomMsg::Hello {
                        name,
                        seat: req,
                        out,
                        ack,
                    } => self.handle_hello(name, req, out, ack),
                    RoomMsg::Disconnect { player: p } => {
                        self.handle_disconnect(p as usize);
                        if p as usize == player {
                            return None;
                        }
                    }
                },
            }
        }
    }

    fn bot_decision(&mut self, action: &Action, seat: Seat) -> Decision {
        let player = self.game.player_at(seat);
        let view = self.game.view(seat);
        match &mut self.occupants[player] {
            Occupant::Bot { agent, .. } => agent_decide(agent.as_mut(), action, &view),
            _ => unreachable!("bot_decision on a non-bot seat"),
        }
    }

    fn fallback_decision(&mut self, action: &Action, seat: Seat) -> Decision {
        let view = self.game.view(seat);
        agent_decide(&mut self.fallback, action, &view)
    }

    /// Lets seated bots observe how the hand ended, from their own seat's point of
    /// view (stateful agents can learn).
    fn notify_hand_end(&mut self) {
        for player in 0..4 {
            let seat = self.game.seat_of(player);
            let view = self.game.view(seat);
            let result = self.game.hand_result(seat);
            if let Occupant::Bot { agent, .. } = &mut self.occupants[player] {
                agent.observe_hand_end(&view, &result);
            }
        }
    }

    // --- Seating -------------------------------------------------------------

    fn handle_hello(
        &mut self,
        name: String,
        requested: Option<Player>,
        out: mpsc::UnboundedSender<ServerMsg>,
        ack: oneshot::Sender<Option<Player>>,
    ) {
        let Some(player) = self.pick_seat(requested) else {
            let _ = ack.send(None);
            return;
        };
        self.occupants[player] = Occupant::Human {
            name,
            out: out.clone(),
        };
        // Fill any still-open seats with bots so a match can run.
        self.fill_bots();
        let _ = ack.send(Some(player as u8));
        let _ = out.send(ServerMsg::Joined {
            players: self.roster(),
            your_seat: player as u8,
            first_dealer: self.game.dealer() as u8,
        });
        // Hand the joiner a full snapshot so a mid-hand join renders correctly.
        let _ = out.send(ServerMsg::Sync {
            view: snapshot(&self.game, player as u8),
        });
    }

    /// Picks a table position for a joining human: the requested one if free, else
    /// any open position, else any bot position (the human takes over). `None` if
    /// the table is all humans.
    fn pick_seat(&self, requested: Option<Player>) -> Option<usize> {
        let available = |p: usize| !matches!(self.occupants[p], Occupant::Human { .. });
        if let Some(req) = requested {
            let p = req as usize;
            if p < 4 && available(p) {
                return Some(p);
            }
        }
        (0..4)
            .find(|&p| matches!(self.occupants[p], Occupant::Open))
            .or_else(|| (0..4).find(|&p| available(p)))
    }

    fn handle_disconnect(&mut self, player: usize) {
        if matches!(self.occupants[player], Occupant::Human { .. }) {
            self.occupants[player] = bot_for(player);
        }
    }

    fn fill_bots(&mut self) {
        for player in 0..4 {
            if matches!(self.occupants[player], Occupant::Open) {
                self.occupants[player] = bot_for(player);
            }
        }
    }

    fn roster(&self) -> Vec<SeatedPlayer> {
        (0..4)
            .filter_map(|player| match &self.occupants[player] {
                Occupant::Human { name, .. } => Some(SeatedPlayer {
                    seat: player as u8,
                    name: name.clone(),
                    bot: false,
                }),
                Occupant::Bot { name, .. } => Some(SeatedPlayer {
                    seat: player as u8,
                    name: name.clone(),
                    bot: true,
                }),
                Occupant::Open => None,
            })
            .collect()
    }

    fn is_human(&self, player: usize) -> bool {
        matches!(self.occupants[player], Occupant::Human { .. })
    }

    // --- Sending -------------------------------------------------------------

    fn broadcast_deal(&self) {
        let dealer = self.game.dealer() as u8;
        let up_card = self.game.up_card();
        for player in 0..4 {
            if let Occupant::Human { out, .. } = &self.occupants[player] {
                let seat = self.game.seat_of(player);
                let _ = out.send(ServerMsg::Deal {
                    dealer,
                    hand: self.game.hand(seat).to_vec(),
                    up_card,
                });
            }
        }
    }

    /// Sends each seated human the just-completed hand's result, told from their
    /// own seat's point of view.
    fn broadcast_hand_complete(&self) {
        for player in 0..4 {
            if let Occupant::Human { out, .. } = &self.occupants[player] {
                let seat = self.game.seat_of(player);
                let _ = out.send(ServerMsg::HandComplete {
                    result: self.game.hand_result(seat),
                });
            }
        }
    }

    /// Tells everyone whose turn it is; the active human also gets its legal
    /// plays (a card's legality can reveal a void, so only that seat sees them).
    fn broadcast_awaiting(&self, action: &Action, active: Seat) {
        let hint = hint_for(action, &self.game);
        let legal = match action {
            Action::Play { legal, .. } => Some(legal.clone()),
            _ => None,
        };
        let active_player = self.game.player_at(active) as u8;
        for player in 0..4 {
            if let Occupant::Human { out, .. } = &self.occupants[player] {
                let legal = if player as u8 == active_player {
                    legal.clone()
                } else {
                    None
                };
                let _ = out.send(ServerMsg::Awaiting {
                    player: active_player,
                    hint: hint.clone(),
                    legal,
                });
            }
        }
    }

    fn broadcast(&self, msg: ServerMsg) {
        for player in 0..4 {
            if let Occupant::Human { out, .. } = &self.occupants[player] {
                let _ = out.send(msg.clone());
            }
        }
    }

    fn send_to(&self, player: usize, msg: ServerMsg) {
        if let Occupant::Human { out, .. } = &self.occupants[player] {
            let _ = out.send(msg);
        }
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

fn bot_for(player: usize) -> Occupant {
    Occupant::Bot {
        name: format!("Bot {}", player_name(player)),
        agent: Box::new(HeuristicAgent::new()),
    }
}

/// The conventional name of a fixed table position (`0` = North … `3` = West).
fn player_name(player: usize) -> &'static str {
    ["North", "East", "South", "West"][player]
}
