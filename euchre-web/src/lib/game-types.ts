//! Small UI-only types used by the game store.

/**
 * The lifecycle of the websocket connection, from the UI's point of view.
 *
 *   * `lobby` — connected and arranging seats at the table.
 *   * `playing` — a match is in progress.
 *   * `closed` — the socket dropped during a match (the board stays up to show
 *     the disconnect notice).
 */
export type ConnStatus = 'idle' | 'connecting' | 'lobby' | 'playing' | 'closed' | 'error';

/** A short, auto-expiring caption shown over a seat (e.g. "pass", "orders up ♥"). */
export interface ActionBubble {
  text: string;
  /** Bumped per bubble so a stale clear-timer never wipes a newer message. */
  key: number;
}
