<script lang="ts">
  import { GameStore } from './lib/game.svelte';
  import { setGame } from './lib/context';
  import StartScreen from './components/StartScreen.svelte';
  import Table from './components/Table.svelte';

  const game = new GameStore();
  setGame(game);

  // Show the table once seated (and keep showing it if the socket later drops,
  // so the disconnect notice has somewhere to live).
  const seated = $derived(game.status === 'joined' || game.status === 'closed');
</script>

<main>
  {#if seated}
    <Table />
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
