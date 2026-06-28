<script lang="ts">
  import { getGame } from '../lib/context';
  import type { Player } from '../lib/protocol';
  import { relativePosition, FLY_OFFSET } from '../lib/seating';
  import PlayingCard from './PlayingCard.svelte';
  import { fly } from 'svelte/transition';
  import { cubicOut, cubicIn } from 'svelte/easing';

  const game = getGame();

  // `mySeat` is always set once the table view is shown.
  const pos = (seat: Player) => relativePosition(game.mySeat!, seat);
  const winnerPos = $derived(
    game.collecting ? relativePosition(game.mySeat!, game.collecting.winner) : null,
  );
</script>

<div class="trick">
  <!-- The trick in progress: each card flies in from its player's seat. -->
  {#each game.table as play (play.seat + play.card)}
    {@const p = pos(play.seat)}
    <div
      class="slot {p}"
      in:fly={{ x: FLY_OFFSET[p].x, y: FLY_OFFSET[p].y, duration: 360, easing: cubicOut }}
    >
      <PlayingCard code={play.card} size="md" trump={game.trump} />
    </div>
  {/each}

  <!-- A finished trick being swept toward its winner. -->
  {#if game.collecting}
    {#each game.collecting.plays as play (play.seat + play.card)}
      {@const p = pos(play.seat)}
      <div
        class="slot {p}"
        out:fly={{
          x: winnerPos ? FLY_OFFSET[winnerPos].x : 0,
          y: winnerPos ? FLY_OFFSET[winnerPos].y : 0,
          duration: 420,
          easing: cubicIn,
        }}
      >
        <PlayingCard code={play.card} size="md" trump={game.trump} />
      </div>
    {/each}
  {/if}
</div>

<style>
  .trick {
    position: relative;
    width: 200px;
    height: 200px;
  }
  .slot {
    position: absolute;
    left: 50%;
    top: 50%;
  }
  .slot.bottom {
    transform: translate(-50%, 18px);
  }
  .slot.top {
    transform: translate(-50%, calc(-100% - 18px));
  }
  .slot.left {
    transform: translate(calc(-100% - 18px), -50%);
  }
  .slot.right {
    transform: translate(18px, -50%);
  }
</style>
