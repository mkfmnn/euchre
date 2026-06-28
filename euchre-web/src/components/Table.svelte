<script lang="ts">
  import { getGame } from '../lib/context';
  import { playersByPosition } from '../lib/seating';
  import type { TeamId } from '../lib/protocol';
  import Seat from './Seat.svelte';
  import OpponentHand from './OpponentHand.svelte';
  import Hand from './Hand.svelte';
  import TrickArea from './TrickArea.svelte';
  import Controls from './Controls.svelte';
  import Scoreboard from './Scoreboard.svelte';
  import PlayingCard from './PlayingCard.svelte';
  import { fade, scale } from 'svelte/transition';

  const game = getGame();
  const seats = $derived(playersByPosition(game.mySeat!));
  const teamName = (t: TeamId) => (t === 0 ? 'North / South' : 'East / West');
</script>

<div class="table">
  <Scoreboard />

  <div class="pos top">
    <Seat seat={seats.top} />
    <OpponentHand seat={seats.top} pos="top" />
  </div>

  <div class="pos left">
    <OpponentHand seat={seats.left} pos="left" />
    <Seat seat={seats.left} />
  </div>

  <div class="pos right">
    <OpponentHand seat={seats.right} pos="right" />
    <Seat seat={seats.right} />
  </div>

  <div class="center">
    {#if game.upCardLive && game.upCard}
      <div class="upcard" transition:scale={{ duration: 200 }}>
        <PlayingCard code={game.upCard} size="md" />
        <span class="up-label">up-card</span>
      </div>
    {/if}
    <TrickArea />
  </div>

  <div class="pos bottom">
    <Hand />
    <Seat seat={seats.bottom} />
  </div>

  <div class="dock">
    <Controls />
  </div>

  {#if game.banner}
    <div class="banner" transition:fade={{ duration: 200 }}>{game.banner}</div>
  {/if}
  {#if game.toast}
    <div class="toast" transition:fade={{ duration: 150 }}>{game.toast}</div>
  {/if}

  {#if game.status === 'closed'}
    <div class="overlay">
      <div class="panel">
        <h2>Disconnected</h2>
        <p>The connection to the server closed.</p>
      </div>
    </div>
  {/if}

  {#if game.gameOver}
    <div class="overlay" transition:fade={{ duration: 250 }}>
      <div class="panel">
        <h2>{teamName(game.gameOver.winner)} wins!</h2>
        <p class="final">
          N/S {game.gameOver.scores.north_south} — {game.gameOver.scores.east_west} E/W
        </p>
        <button onclick={() => (game.gameOver = null)}>Play again</button>
      </div>
    </div>
  {/if}
</div>

<style>
  .table {
    position: relative;
    width: min(96vw, 1000px);
    height: min(94vh, 760px);
    border-radius: 28px;
    background:
      radial-gradient(ellipse at center, #2a8a5b 0%, #1c6e46 60%, #155737 100%);
    box-shadow:
      inset 0 0 0 10px rgba(0, 0, 0, 0.18),
      0 18px 50px rgba(0, 0, 0, 0.45);
  }

  .pos {
    position: absolute;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }
  .pos.top {
    top: 14px;
    left: 50%;
    transform: translateX(-50%);
  }
  .pos.bottom {
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
  }
  .pos.left {
    left: 18px;
    top: 50%;
    transform: translateY(-50%);
  }
  .pos.right {
    right: 18px;
    top: 50%;
    transform: translateY(-50%);
  }

  .center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }
  .upcard {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    margin-bottom: 4px;
  }
  .up-label {
    font-size: 0.72rem;
    opacity: 0.7;
  }

  .dock {
    position: absolute;
    bottom: 180px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 6;
  }

  .banner {
    position: absolute;
    top: 18%;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(8, 32, 21, 0.95);
    border: 1px solid #e7b53d;
    border-radius: 12px;
    padding: 10px 20px;
    font-size: 1.1rem;
    font-weight: 600;
    white-space: nowrap;
    z-index: 7;
  }
  .toast {
    position: absolute;
    bottom: 14px;
    right: 14px;
    background: rgba(60, 20, 20, 0.95);
    border: 1px solid #c0563b;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.85rem;
    z-index: 7;
  }

  .overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.55);
    border-radius: 28px;
    z-index: 10;
  }
  .panel {
    text-align: center;
    background: #0c2417;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 28px 40px;
    box-shadow: 0 16px 50px rgba(0, 0, 0, 0.5);
  }
  .panel h2 {
    margin: 0 0 6px;
  }
  .panel .final {
    font-size: 1.2rem;
    margin: 0 0 18px;
  }
  .panel button {
    font: inherit;
    font-weight: 600;
    padding: 10px 22px;
    border: none;
    border-radius: 9px;
    background: #e7b53d;
    color: #14361f;
    cursor: pointer;
  }
</style>
