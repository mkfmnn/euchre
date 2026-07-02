<script lang="ts">
  import { getGame } from '../lib/context';
  import type { CardCode } from '../lib/protocol';
  import PlayingCard from './PlayingCard.svelte';
  import { flip } from 'svelte/animate';

  const game = getGame();

  function selectable(code: CardCode): boolean {
    if (game.awaitingDiscard) return true;
    if (game.awaitingPlay) return game.legal ? game.legal.includes(code) : true;
    return false;
  }

  function dimmed(code: CardCode): boolean {
    return game.awaitingPlay && game.legal !== null && !game.legal.includes(code);
  }

  function pick(code: CardCode): void {
    if (game.awaitingDiscard) game.discardCard(code);
    else if (game.awaitingPlay && selectable(code)) game.playCard(code);
  }
</script>

<div class="my-hand">
  {#each game.hand as code (code)}
    <!-- The assist ring and score live on the slot wrapper so they sit clear of
         the card's own button hover/outline styling. -->
    <div
      class="slot"
      class:recommended={game.isRecommendedCard(code)}
      title={game.cardTip(code)}
      animate:flip={{ duration: 260 }}
    >
      <PlayingCard
        {code}
        size="lg"
        trump={game.trump}
        selectable={selectable(code)}
        dim={dimmed(code)}
        onpick={() => pick(code)}
      />
    </div>
  {/each}
</div>

<style>
  .my-hand {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    min-height: 110px;
  }
  .slot {
    margin-left: -14px;
    border-radius: 10px;
  }
  .slot:first-child {
    margin-left: 0;
  }
  /* The assist agent's recommended card: a green ring, lifted above its
     neighbours so the overlap never clips it. */
  .slot.recommended {
    position: relative;
    z-index: 5;
    outline: 3px solid #2fbf71;
    outline-offset: 2px;
    box-shadow: 0 0 10px rgba(47, 191, 113, 0.7);
  }
</style>
