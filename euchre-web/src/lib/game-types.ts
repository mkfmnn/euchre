//! Small UI-only types used by the game store.

/** The lifecycle of the websocket connection, from the UI's point of view. */
export type ConnStatus = 'idle' | 'connecting' | 'joined' | 'closed' | 'error';

/** A short, auto-expiring caption shown over a seat (e.g. "pass", "orders up ♥"). */
export interface ActionBubble {
  text: string;
  /** Bumped per bubble so a stale clear-timer never wipes a newer message. */
  key: number;
}
