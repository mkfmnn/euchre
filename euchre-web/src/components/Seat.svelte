<script lang="ts">
  import { getGame } from '../lib/context';
  import type { Seat } from '../lib/protocol';
  import { fade } from 'svelte/transition';

  let { seat }: { seat: Seat } = $props();
  const game = getGame();

  const player = $derived(game.players[seat]);
  const name = $derived(player?.name ?? seat);
  const isDealer = $derived(game.dealer === seat);
  const isMaker = $derived(game.maker === seat);
  const isTurn = $derived(game.whoseTurn === seat);
  const isMe = $derived(game.mySeat === seat);
  const tricks = $derived(game.tricksWon[seat]);
  const sittingOut = $derived(game.sittingOut === seat);
  const bubble = $derived(game.bubbles[seat]);
</script>

<div class="seat" class:turn={isTurn}>
  <div class="line">
    <span class="name" class:me={isMe}>{name}</span>
    {#if isDealer}<span class="chip" title="dealer">D</span>{/if}
    {#if isMaker}<span class="chip maker" title="maker">M</span>{/if}
  </div>
  <div class="line sub">
    <span class="tricks">{tricks} {tricks === 1 ? 'trick' : 'tricks'}</span>
    {#if sittingOut}<span class="out">sitting out</span>{/if}
  </div>
  {#if bubble}
    {#key bubble.key}
      <div class="bubble" in:fade={{ duration: 120 }} out:fade={{ duration: 300 }}>
        {bubble.text}
      </div>
    {/key}
  {/if}
</div>

<style>
  .seat {
    position: relative;
    text-align: center;
    padding: 4px 12px;
    border-radius: 10px;
    transition: background 0.2s ease;
  }
  .seat.turn {
    background: rgba(231, 181, 61, 0.18);
    box-shadow: 0 0 0 1px rgba(231, 181, 61, 0.55);
  }
  .line {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }
  .name {
    font-weight: 600;
  }
  .name.me {
    color: #e7b53d;
  }
  .sub {
    font-size: 0.78rem;
    opacity: 0.75;
    gap: 8px;
  }
  .chip {
    font-size: 0.7rem;
    font-weight: 700;
    width: 17px;
    height: 17px;
    line-height: 17px;
    border-radius: 50%;
    background: #f3f4f6;
    color: #14361f;
  }
  .chip.maker {
    background: #e7b53d;
  }
  .out {
    color: #f0a868;
  }
  .bubble {
    position: absolute;
    left: 50%;
    bottom: calc(100% + 4px);
    transform: translateX(-50%);
    white-space: nowrap;
    background: #11261a;
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 0.82rem;
    pointer-events: none;
  }
</style>
