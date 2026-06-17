import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// A purely static front end: the only "backend" it talks to is the
// euchre-server websocket. `vite build` emits a folder of static assets.
export default defineConfig({
  plugins: [svelte()],
});
