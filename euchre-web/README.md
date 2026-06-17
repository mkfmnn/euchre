# euchre-web

A minimal web front end for playing Euchre, built with **Svelte 5**, **Vite**,
and **TypeScript 6**. It is a fully static app: the only thing it talks to is the
`euchre-server` crate, over a single websocket.

This first version plays one human (you) against three server-side bots and runs
a single match to its conclusion.

## Running it

Start the game server (from the workspace root):

```bash
cargo run -p euchre-server          # listens on ws://127.0.0.1:8080/ws
```

Then, in this directory:

```bash
npm install
npm run dev                         # Vite dev server, prints a localhost URL
```

Open the printed URL, confirm the server address (defaults to
`ws://<host>:8080/ws`), and sit down. You are seated at the bottom of the table;
your partner is across from you and the two opponents are to the sides.

Other scripts:

```bash
npm run build       # type-checked production build into dist/
npm run preview     # serve the built dist/ locally
npm run check       # svelte-check (type + template checking) only
```

`npm run build` runs `svelte-check` first, so a build only succeeds when the
types and templates are clean.

## How it talks to the server

The wire protocol is mirrored in [`src/lib/protocol.ts`](src/lib/protocol.ts),
matching `euchre-server`'s `protocol.rs` exactly: tagged JSON messages
(`"type"` / `"kind"` in `SCREAMING_SNAKE_CASE`), two-letter card codes (`"JS"`),
and suit/seat/team names spelled as their Rust variants.

The protocol is **event-sourced**, and so is the client. After the `HELLO`
handshake the server sends a private `DEAL`, then a stream of `AWAITING`,
`UPDATE`, `TRICK_WON`, `HAND_COMPLETE`, and `GAME_OVER` events. The store in
[`src/lib/game.svelte.ts`](src/lib/game.svelte.ts) folds these into reactive UI
state. Hidden information the wire deliberately omits is reconstructed locally —
most notably, the dealer's picked-up up-card is folded into the hand exactly when
the server asks that seat to discard.

## Layout of the code

- `src/lib/protocol.ts` — TypeScript types for every wire message.
- `src/lib/cards.ts` — card parsing and the trump/bower comparison rules, ported
  from `euchre-interface`, used to sort hands and highlight trump.
- `src/lib/seating.ts` — maps absolute seats to on-screen positions (you at the
  bottom) and the fly-in/out directions for animations.
- `src/lib/game.svelte.ts` — the websocket client and event-sourced game state.
- `src/components/` — the Svelte views: the table, hands, trick area, bidding
  controls, scoreboard, and start screen.

## Animations

Two clarifying animations, both built on Svelte's `fly` transition:

- a played card flies from its player's seat onto the table;
- a completed trick is swept off the table toward the seat that won it.

## Pacing

The server and its bots emit events far faster than a human can follow, so the
client paces them: incoming messages go through a small render queue
(`game.svelte.ts`) that applies them one at a time, in order, with legible gaps.
Pacing covers every action — bids, passes, discards, and card plays alike. The
delays are *minimums* — time a message already spent in flight counts toward
them, so a slow reply shows at once while a fast one waits out the remainder.
Two constants are the knobs (your own actions always render the instant you make
them):

- `ACTION_GAP_MS` (500ms) — minimum gap between consecutive actions (bid, pass, discard, play);
- `TRICK_LINGER_MS` (1000ms) — how long a finished trick rests before being swept.
