<script lang="ts">
  import { getGame } from '../lib/context';
  import { PLAYER_LABELS } from '../lib/seating';
  import type { Player } from '../lib/protocol';

  const game = getGame();

  const seated = $derived(game.mySeat !== null);
  const allSeated = $derived(game.seats.every((s) => s.type !== 'Empty'));

  // Local countdown feedback once the table fills; the server's START_GAME is
  // what actually begins the match.
  let countdown = $state<number | null>(null);
  $effect(() => {
    if (!allSeated) {
      countdown = null;
      return;
    }
    countdown = 5;
    const timer = setInterval(() => {
      countdown = countdown !== null && countdown > 0 ? countdown - 1 : 0;
    }, 1000);
    return () => clearInterval(timer);
  });

  // The four seats in a fixed diamond, regardless of where the viewer sits.
  const layout: { seat: Player; pos: 'top' | 'right' | 'bottom' | 'left' }[] = [
    { seat: 0, pos: 'top' },
    { seat: 1, pos: 'right' },
    { seat: 2, pos: 'bottom' },
    { seat: 3, pos: 'left' },
  ];
</script>

<div class="lobby">
  <header>
    <h1>Euchre</h1>
    <p class="code">Table <strong>{game.tableCode}</strong></p>
  </header>

  <div class="felt">
    {#each layout as { seat, pos } (seat)}
      {@const info = game.seats[seat]}
      <div class="pos {pos}" class:me={seat === game.mySeat}>
        <div class="label">{PLAYER_LABELS[seat]}</div>
        {#if info.type === 'Empty'}
          {#if seated}
            <button onclick={() => game.addBot(seat)}>Add Bot</button>
          {:else}
            <button onclick={() => game.sit(seat)}>Sit Here</button>
          {/if}
        {:else}
          <div class="occupant">
            <span class="name">{info.name}</span>
            {#if info.type === 'Bot' || seat === game.mySeat}
              <button class="x" title="Remove" onclick={() => game.vacate(seat)}>✕</button>
            {/if}
          </div>
        {/if}
      </div>
    {/each}

    <div class="center">
      {#if allSeated}
        <p class="starting">
          Starting{countdown ? ` in ${countdown}…` : '…'}
        </p>
      {:else if seated}
        <p class="hint">Add bots or wait for players to fill the table.</p>
      {:else}
        <p class="hint">Pick a seat to sit down.</p>
      {/if}
    </div>
  </div>

  {#if game.error}
    <p class="err">{game.error}</p>
  {/if}
</div>

<style>
  .lobby {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 18px;
    width: min(94vw, 560px);
  }
  header {
    text-align: center;
  }
  h1 {
    margin: 0;
    font-size: 2rem;
    letter-spacing: 0.04em;
  }
  .code {
    margin: 4px 0 0;
    opacity: 0.85;
  }
  .code strong {
    color: #e7b53d;
    letter-spacing: 0.18em;
    font-size: 1.2rem;
  }
  .felt {
    position: relative;
    width: min(90vw, 520px);
    aspect-ratio: 1 / 1;
    background: radial-gradient(circle at 50% 45%, #1e6b45, #11402a 70%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 24px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
  }
  .pos {
    position: absolute;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    min-width: 96px;
  }
  .pos.top {
    top: 16px;
    left: 50%;
    transform: translateX(-50%);
  }
  .pos.bottom {
    bottom: 16px;
    left: 50%;
    transform: translateX(-50%);
  }
  .pos.left {
    left: 16px;
    top: 50%;
    transform: translateY(-50%);
  }
  .pos.right {
    right: 16px;
    top: 50%;
    transform: translateY(-50%);
  }
  .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.6;
  }
  .pos.me .name {
    color: #e7b53d;
  }
  .occupant {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 0, 0, 0.28);
    padding: 6px 10px;
    border-radius: 10px;
  }
  .name {
    font-weight: 600;
  }
  button {
    font: inherit;
    font-weight: 600;
    padding: 8px 14px;
    border-radius: 9px;
    border: none;
    background: #e7b53d;
    color: #14361f;
    cursor: pointer;
  }
  button.x {
    padding: 0;
    width: 20px;
    height: 20px;
    line-height: 20px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.18);
    color: inherit;
    font-size: 0.7rem;
  }
  .center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 60%;
    text-align: center;
  }
  .starting {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e7b53d;
    margin: 0;
  }
  .hint {
    margin: 0;
    opacity: 0.7;
    font-size: 0.9rem;
  }
  .err {
    margin: 0;
    color: #f0a868;
    font-size: 0.9rem;
  }
</style>
