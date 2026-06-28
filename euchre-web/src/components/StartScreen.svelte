<script lang="ts">
  import { getGame } from '../lib/context';

  const game = getGame();

  const defaultUrl = `ws://${location.hostname || '127.0.0.1'}:8080/ws`;
  let name = $state('You');
  let url = $state(defaultUrl);
  /** Whether the join-by-code field is showing. */
  let joining = $state(false);
  let code = $state('');

  const connecting = $derived(game.status === 'connecting');

  function createTable(): void {
    game.connect(url.trim(), name, null);
  }
  function joinTable(event: SubmitEvent): void {
    event.preventDefault();
    if (code.trim().length === 0) return;
    game.connect(url.trim(), name, code.trim());
  }
</script>

<div class="start">
  <h1>Euchre</h1>
  <p class="tagline">Create a table or join one by code.</p>

  <label>
    Name
    <input bind:value={name} maxlength="20" autocomplete="off" />
  </label>
  <label>
    Server
    <input bind:value={url} autocomplete="off" spellcheck="false" />
  </label>

  {#if joining}
    <form class="join" onsubmit={joinTable}>
      <label>
        Table code
        <input
          bind:value={code}
          maxlength="4"
          inputmode="numeric"
          placeholder="0000"
          autocomplete="off"
        />
      </label>
      <div class="row">
        <button type="button" class="ghost" onclick={() => (joining = false)} disabled={connecting}>
          Back
        </button>
        <button type="submit" disabled={connecting || code.trim().length === 0}>
          {connecting ? 'Joining…' : 'Join'}
        </button>
      </div>
    </form>
  {:else}
    <div class="row">
      <button type="button" onclick={createTable} disabled={connecting}>
        {connecting ? 'Connecting…' : 'Create Table'}
      </button>
      <button type="button" class="ghost" onclick={() => (joining = true)} disabled={connecting}>
        Join Table
      </button>
    </div>
  {/if}

  {#if game.error}
    <p class="err">{game.error}</p>
  {/if}

  <p class="hint">Run the server first: <code>cargo run -p euchre-server</code></p>
</div>

<style>
  .start {
    display: flex;
    flex-direction: column;
    gap: 14px;
    width: min(92vw, 360px);
    background: rgba(8, 32, 21, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
  }
  h1 {
    margin: 0;
    font-size: 2rem;
    letter-spacing: 0.04em;
  }
  .tagline {
    margin: -6px 0 6px;
    opacity: 0.75;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 0.85rem;
    opacity: 0.9;
  }
  input {
    font: inherit;
    padding: 9px 11px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    background: rgba(0, 0, 0, 0.25);
    color: inherit;
  }
  button {
    font: inherit;
    font-weight: 600;
    margin-top: 4px;
    padding: 11px;
    border-radius: 9px;
    border: none;
    background: #e7b53d;
    color: #14361f;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .ghost {
    background: transparent;
    color: inherit;
    border: 1px solid rgba(255, 255, 255, 0.3);
  }
  .row {
    display: flex;
    gap: 10px;
  }
  .row button {
    flex: 1;
  }
  .join {
    display: contents;
  }
  .err {
    margin: 0;
    color: #f0a868;
    font-size: 0.9rem;
  }
  .hint {
    margin: 0;
    font-size: 0.78rem;
    opacity: 0.6;
  }
  code {
    background: rgba(0, 0, 0, 0.3);
    padding: 1px 5px;
    border-radius: 4px;
  }
</style>
