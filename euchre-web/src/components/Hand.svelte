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
    <div class="slot" animate:flip={{ duration: 260 }}>
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
  }
  .slot:first-child {
    margin-left: 0;
  }
</style>
