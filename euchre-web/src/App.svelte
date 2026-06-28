<script lang="ts">
  import { GameStore } from './lib/game.svelte';
  import { setGame } from './lib/context';
  import StartScreen from './components/StartScreen.svelte';
  import Lobby from './components/Lobby.svelte';
  import Table from './components/Table.svelte';

  const game = new GameStore();
  setGame(game);

  // The board stays up while a match runs and after the socket drops (so the
  // disconnect notice has somewhere to live); the lobby shows while seating.
  const playing = $derived(game.status === 'playing' || game.status === 'closed');
</script>

<main>
  {#if playing}
    <Table />
  {:else if game.status === 'lobby'}
    <Lobby />
  {:else}
    <StartScreen />
  {/if}
</main>

<style>
  main {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
  }
</style>
