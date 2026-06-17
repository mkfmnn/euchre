//! Card parsing and the trump-aware comparison rules, ported from the engine's
//! `euchre-interface::card`. The bower rules live here so the UI can sort hands
//! and highlight trump the same way the server ranks tricks.

import type { Suit, CardCode } from './protocol';

export type Rank = '9' | '10' | 'J' | 'Q' | 'K' | 'A';

export interface Card {
  rank: Rank;
  suit: Suit;
  /** The original two-letter code, kept so cards round-trip to the wire. */
  code: CardCode;
}

const RANK_FROM_CODE: Record<string, Rank> = {
  '9': '9',
  T: '10',
  J: 'J',
  Q: 'Q',
  K: 'K',
  A: 'A',
};

const SUIT_FROM_CODE: Record<string, Suit> = {
  C: 'Clubs',
  D: 'Diamonds',
  H: 'Hearts',
  S: 'Spades',
};

/** Natural rank order (matches the engine's `Rank as u32`): 9 lowest, A highest. */
const RANK_VALUE: Record<Rank, number> = { '9': 0, '10': 1, J: 2, Q: 3, K: 4, A: 5 };

/** Parses a two-letter wire code (`"JS"`) into a structured card. */
export function parseCard(code: CardCode): Card {
  return { rank: RANK_FROM_CODE[code[0]], suit: SUIT_FROM_CODE[code[1]], code };
}

export const SUIT_SYMBOL: Record<Suit, string> = {
  Clubs: '♣',
  Diamonds: '♦',
  Hearts: '♥',
  Spades: '♠',
};

export const SUIT_NAME: Record<Suit, string> = {
  Clubs: 'Clubs',
  Diamonds: 'Diamonds',
  Hearts: 'Hearts',
  Spades: 'Spades',
};

/** Whether a suit is rendered in red. */
export function isRed(suit: Suit): boolean {
  return suit === 'Diamonds' || suit === 'Hearts';
}

/** The other suit of the same color — the left bower's printed suit. */
export function sameColor(suit: Suit): Suit {
  switch (suit) {
    case 'Clubs':
      return 'Spades';
    case 'Spades':
      return 'Clubs';
    case 'Diamonds':
      return 'Hearts';
    case 'Hearts':
      return 'Diamonds';
  }
}

export function isRightBower(card: Card, trump: Suit): boolean {
  return card.rank === 'J' && card.suit === trump;
}

export function isLeftBower(card: Card, trump: Suit): boolean {
  return card.rank === 'J' && card.suit === sameColor(trump);
}

/** Whether the card counts as trump (the left bower does, despite its print). */
export function isTrump(card: Card, trump: Suit): boolean {
  return card.suit === trump || isLeftBower(card, trump);
}

/** The suit a card follows; the left bower follows trump, not its printed suit. */
export function effectiveSuit(card: Card, trump: Suit): Suit {
  return isLeftBower(card, trump) ? trump : card.suit;
}

/** Trump-aware strength within a trick; higher beats lower (see the engine). */
export function trumpStrength(card: Card, trump: Suit, led: Suit): number {
  if (isRightBower(card, trump)) return 1000;
  if (isLeftBower(card, trump)) return 999;
  if (card.suit === trump) return 100 + RANK_VALUE[card.rank];
  if (effectiveSuit(card, trump) === led) return 10 + RANK_VALUE[card.rank];
  return RANK_VALUE[card.rank];
}

// A display order for the four suits that alternates colors, so a fanned hand
// never sits two same-colored suits side by side.
const DISPLAY_ORDER: Suit[] = ['Clubs', 'Hearts', 'Spades', 'Diamonds'];

/**
 * Sorts a hand for display: when trump is known, trump (with the bowers on top)
 * comes first, then the off-suits high-to-low; before trump is set, simply group
 * by suit, high cards first.
 */
export function sortHand(codes: CardCode[], trump: Suit | null): CardCode[] {
  const groupKey = (c: Card): number => {
    if (trump && isTrump(c, trump)) return -1;
    return DISPLAY_ORDER.indexOf(c.suit);
  };
  const rankKey = (c: Card): number => {
    if (trump && isTrump(c, trump)) return trumpStrength(c, trump, trump);
    return RANK_VALUE[c.rank];
  };
  return codes
    .map(parseCard)
    .sort((a, b) => groupKey(a) - groupKey(b) || rankKey(b) - rankKey(a))
    .map((c) => c.code);
}
