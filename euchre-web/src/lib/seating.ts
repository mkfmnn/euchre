//! Mapping absolute seats to on-screen positions, relative to the viewer.
//!
//! The local player always sits at the bottom; the rest fall out from the
//! clockwise order N → E → S → W. Each position also carries a fly offset — the
//! direction a card travels from when played from that seat (and toward when a
//! trick is swept to its winner).

import type { Seat } from './protocol';

/** The four on-screen positions, viewer at the bottom. */
export type RelPos = 'bottom' | 'left' | 'top' | 'right';

const CLOCKWISE: Seat[] = ['North', 'East', 'South', 'West'];
const POSITIONS: RelPos[] = ['bottom', 'left', 'top', 'right'];

/** Where `seat` should be drawn, given the viewer occupies `me`. */
export function relativePosition(me: Seat, seat: Seat): RelPos {
  const delta = (CLOCKWISE.indexOf(seat) - CLOCKWISE.indexOf(me) + 4) % 4;
  return POSITIONS[delta];
}

/** The seat occupying each on-screen position for viewer `me`. */
export function seatsByPosition(me: Seat): Record<RelPos, Seat> {
  const out = {} as Record<RelPos, Seat>;
  for (const seat of CLOCKWISE) out[relativePosition(me, seat)] = seat;
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
