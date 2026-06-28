//! Mapping fixed table positions to on-screen positions, relative to the viewer.
//!
//! The local player always sits at the bottom; the rest fall out from the fixed
//! clockwise order N → E → S → W (player `0` → `1` → `2` → `3`). Each position
//! also carries a fly offset — the direction a card travels from when played
//! from that seat (and toward when a trick is swept to its winner).

import type { Player } from './protocol';

/** Conventional name of each fixed table position, used as a name fallback. */
export const PLAYER_LABELS = ['North', 'East', 'South', 'West'];

/** The four on-screen positions, viewer at the bottom. */
export type RelPos = 'bottom' | 'left' | 'top' | 'right';

const POSITIONS: RelPos[] = ['bottom', 'left', 'top', 'right'];

/** Where `player` should be drawn, given the viewer occupies `me`. */
export function relativePosition(me: Player, player: Player): RelPos {
  // Players are numbered clockwise, so the on-screen position is just the
  // clockwise distance from the viewer.
  return POSITIONS[(player - me + 4) % 4];
}

/** The player occupying each on-screen position for viewer `me`. */
export function playersByPosition(me: Player): Record<RelPos, Player> {
  const out = {} as Record<RelPos, Player>;
  for (let p = 0; p < 4; p++) out[relativePosition(me, p)] = p;
  return out;
}

/**
 * The travel offset (in px) for a card belonging to each position: cards fly in
 * from this offset toward their resting slot, and a swept trick flies out toward
 * the winner's offset.
 */
export const FLY_OFFSET: Record<RelPos, { x: number; y: number }> = {
  bottom: { x: 0, y: 220 },
  left: { x: -300, y: 0 },
  top: { x: 0, y: -220 },
  right: { x: 300, y: 0 },
};
