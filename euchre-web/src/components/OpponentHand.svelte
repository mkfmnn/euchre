<script lang="ts">
  import { getGame } from '../lib/context';
  import type { Seat } from '../lib/protocol';
  import type { RelPos } from '../lib/seating';
  import PlayingCard from './PlayingCard.svelte';
  import { flip } from 'svelte/animate';

  let { seat, pos }: { seat: Seat; pos: RelPos } = $props();
  const game = getGame();

  const backs = $derived(Array.from({ length: game.cardsLeft[seat] }, (_, i) => i));
  const vertical = $derived(pos === 'left' || pos === 'right');
</script>

<div class="opp" class:vertical>
  {#each backs as i (i)}
    <div class="slot" animate:flip={{ duration: 260 }}>
      <PlayingCard faceDown size="sm" />
    </div>
  {/each}
</div>

<style>
  .opp {
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .opp .slot {
    margin-left: -16px;
  }
  .opp .slot:first-child {
    margin-left: 0;
  }
  .opp.vertical {
    flex-direction: column;
  }
  .opp.vertical .slot {
    margin-left: 0;
    margin-top: -34px;
  }
  .opp.vertical .slot:first-child {
    margin-top: 0;
  }
</style>
