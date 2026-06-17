//! A tiny typed wrapper over Svelte context for sharing the single game store.

import { getContext, setContext } from 'svelte';
import type { GameStore } from './game.svelte';

const KEY = Symbol('euchre-game');

export function setGame(game: GameStore): void {
  setContext(KEY, game);
}

export function getGame(): GameStore {
  return getContext(KEY) as GameStore;
}
