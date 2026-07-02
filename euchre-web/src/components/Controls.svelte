<script lang="ts">
  import { getGame } from '../lib/context';
  import type { Suit } from '../lib/protocol';
  import { parseCard, SUIT_SYMBOL, SUIT_NAME, isRed } from '../lib/cards';

  const game = getGame();

  let alone = $state(false);

  const upSuit = $derived(game.upCard ? parseCard(game.upCard).suit : null);
  // Round two: any suit except the turned-down up-card suit may be named.
  const callSuits = $derived(
    (['Clubs', 'Diamonds', 'Hearts', 'Spades'] as Suit[]).filter((s) => s !== upSuit),
  );
</script>

{#if game.awaitingBid && game.hint?.kind === 'BID'}
  <div class="controls">
    {#if game.hint.up}
      <span class="prompt">
        Order up the {upSuit ? SUIT_NAME[upSuit] : ''}
        {#if upSuit}<b class:red={isRed(upSuit)}>{SUIT_SYMBOL[upSuit]}</b>{/if}?
      </span>
      <button
        class="primary"
        class:recommended={upSuit !== null && game.isRecommendedBid(upSuit, false)}
        title={upSuit ? game.bidTip(upSuit, false) : undefined}
        onclick={() => game.orderUp(false)}>Order up</button
      >
      <button
        class="primary"
        class:recommended={upSuit !== null && game.isRecommendedBid(upSuit, true)}
        title={upSuit ? game.bidTip(upSuit, true) : undefined}
        onclick={() => game.orderUp(true)}>Order up alone</button
      >
      {#if game.hint.may_pass}
        <button
          class="ghost"
          class:recommended={game.isRecommendedPass()}
          title={game.passTip()}
          onclick={() => game.pass()}>Pass</button
        >
      {/if}
    {:else}
      <span class="prompt">Name trump:</span>
      {#each callSuits as s (s)}
        <button
          class="suit"
          class:red={isRed(s)}
          class:recommended={game.isRecommendedBid(s, alone)}
          title={game.bidTip(s, alone)}
          onclick={() => game.bid(s, alone)}
        >
          {SUIT_SYMBOL[s]}
        </button>
      {/each}
      <label class="alone"><input type="checkbox" bind:checked={alone} /> alone</label>
      {#if game.hint.may_pass}
        <button
          class="ghost"
          class:recommended={game.isRecommendedPass()}
          title={game.passTip()}
          onclick={() => game.pass()}>Pass</button
        >
      {/if}
    {/if}
  </div>
{:else if game.awaitingDiscard}
  <div class="controls"><span class="prompt">Pick a card to discard.</span></div>
{:else if game.awaitingPlay}
  <div class="controls"><span class="prompt">Your turn — play a card.</span></div>
{/if}

<style>
  .controls {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
    background: rgba(8, 32, 21, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 10px 16px;
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
  }
  .prompt {
    font-weight: 600;
  }
  b.red {
    color: #e8746a;
  }
  button {
    font: inherit;
    border-radius: 8px;
    border: 1px solid transparent;
    padding: 7px 14px;
    cursor: pointer;
  }
  button.primary {
    background: #e7b53d;
    color: #14361f;
    font-weight: 600;
  }
  button.ghost {
    background: transparent;
    border-color: rgba(255, 255, 255, 0.35);
    color: inherit;
  }
  button.suit {
    background: #fbfbf9;
    color: #1b1b1b;
    font-size: 1.3rem;
    line-height: 1;
    padding: 6px 12px;
  }
  button.suit.red {
    color: #c0392b;
  }
  /* The assist agent's recommended bid: a green ring around the button. */
  button.recommended {
    outline: 3px solid #2fbf71;
    outline-offset: 2px;
    box-shadow: 0 0 10px rgba(47, 191, 113, 0.7);
  }
  button:hover {
    filter: brightness(1.08);
  }
  .alone {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.9rem;
  }
</style>
