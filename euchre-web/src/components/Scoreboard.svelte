<script lang="ts">
  import { getGame } from '../lib/context';
  import { SUIT_SYMBOL, isRed } from '../lib/cards';

  const game = getGame();
  const makerName = $derived(game.maker ? (game.players[game.maker]?.name ?? game.maker) : null);
  const myTeam = $derived(
    game.mySeat === 'North' || game.mySeat === 'South' ? 'north_south' : 'east_west',
  );
</script>

<div class="scoreboard">
  <div class="scores">
    <div class="team" class:mine={myTeam === 'north_south'}>
      <span class="label">N/S</span><span class="val">{game.scores.north_south}</span>
    </div>
    <div class="team" class:mine={myTeam === 'east_west'}>
      <span class="label">E/W</span><span class="val">{game.scores.east_west}</span>
    </div>
    <div class="target">to {game.targetScore}</div>
  </div>
  {#if game.trump}
    <div class="contract">
      Trump
      <b class:red={isRed(game.trump)}>{SUIT_SYMBOL[game.trump]}</b>
      · {makerName}{game.alone ? ' (alone)' : ''}
    </div>
  {/if}
</div>

<style>
  .scoreboard {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(8, 32, 21, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 0.9rem;
    z-index: 5;
  }
  .scores {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .team {
    display: flex;
    align-items: baseline;
    gap: 5px;
  }
  .team.mine .label {
    color: #e7b53d;
    font-weight: 700;
  }
  .label {
    opacity: 0.85;
  }
  .val {
    font-size: 1.2rem;
    font-weight: 700;
  }
  .target {
    opacity: 0.6;
    font-size: 0.78rem;
  }
  .contract {
    margin-top: 5px;
    font-size: 0.82rem;
    opacity: 0.9;
  }
  b.red {
    color: #e8746a;
  }
</style>
