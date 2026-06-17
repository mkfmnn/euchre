<script lang="ts">
  import type { Suit, CardCode } from '../lib/protocol';
  import { parseCard, isRed, isTrump, SUIT_SYMBOL } from '../lib/cards';

  let {
    code = null,
    faceDown = false,
    size = 'md',
    trump = null,
    selectable = false,
    dim = false,
    onpick,
  }: {
    code?: CardCode | null;
    faceDown?: boolean;
    size?: 'sm' | 'md' | 'lg';
    trump?: Suit | null;
    selectable?: boolean;
    dim?: boolean;
    onpick?: () => void;
  } = $props();

  const card = $derived(code ? parseCard(code) : null);
  const red = $derived(card ? isRed(card.suit) : false);
  const trumpish = $derived(card && trump ? isTrump(card, trump) : false);
</script>

{#snippet face()}
  {#if faceDown || !card}
    <div class="back"></div>
  {:else}
    <span class="corner">{card.rank}<br />{SUIT_SYMBOL[card.suit]}</span>
    <span class="pip">{SUIT_SYMBOL[card.suit]}</span>
  {/if}
{/snippet}

{#if selectable}
  <button
    type="button"
    class="card {size}"
    class:red
    class:trump={trumpish}
    class:selectable
    onclick={() => onpick?.()}
  >
    {@render face()}
  </button>
{:else}
  <div class="card {size}" class:red class:trump={trumpish} class:dim>
    {@render face()}
  </div>
{/if}

<style>
  .card {
    --w: 54px;
    --h: 78px;
    position: relative;
    width: var(--w);
    height: var(--h);
    border-radius: calc(var(--w) * 0.12);
    background: #fbfbf9;
    color: #1b1b1b;
    border: 1px solid #d6d6cf;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.35);
    padding: 0;
    font-family: inherit;
    user-select: none;
    overflow: hidden;
    display: block;
  }
  .card.sm {
    --w: 38px;
    --h: 55px;
  }
  .card.lg {
    --w: 72px;
    --h: 104px;
  }
  .card.red {
    color: #c0392b;
  }
  .card.trump {
    box-shadow:
      0 0 0 2px #e7b53d,
      0 2px 6px rgba(0, 0, 0, 0.4);
  }
  .card.dim {
    opacity: 0.4;
    filter: grayscale(0.5);
  }

  .corner {
    position: absolute;
    top: 4px;
    left: 5px;
    font-weight: 700;
    line-height: 0.92;
    font-size: calc(var(--w) * 0.26);
    text-align: center;
  }
  .pip {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: calc(var(--w) * 0.6);
    opacity: 0.9;
  }

  .back {
    position: absolute;
    inset: 5px;
    border-radius: calc(var(--w) * 0.08);
    background:
      repeating-linear-gradient(45deg, #2b5a8c 0 6px, #24507e 6px 12px);
    border: 1px solid #16365a;
  }

  button.card.selectable {
    cursor: pointer;
    transition:
      transform 0.12s ease,
      box-shadow 0.12s ease;
  }
  button.card.selectable:hover,
  button.card.selectable:focus-visible {
    transform: translateY(-12px);
    box-shadow: 0 8px 14px rgba(0, 0, 0, 0.45);
    outline: none;
  }
  button.card.selectable.trump:hover,
  button.card.selectable.trump:focus-visible {
    box-shadow:
      0 0 0 2px #e7b53d,
      0 8px 14px rgba(0, 0, 0, 0.45);
  }
</style>
