<script lang="ts">
	import { dataUrl, fmt, sci } from './theme';

	interface Sweep {
		length: number;
		log_mass: number[];
		nodes_by_layer: number[];
		n_nodes: number;
		n_edges: number;
		log_partition: number;
	}

	let sw = $state<Sweep | null>(null);
	let layer = $state(0); // layers swept so far
	let playing = $state(false);
	let failed = $state(false);
	let width = $state(600);

	$effect(() => {
		fetch(dataUrl('viz/model_parts/model_parts.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d) => (sw = d.sweep))
			.catch(() => (failed = true));
	});

	const L = $derived(sw ? sw.length : 0);
	const done = $derived(sw !== null && layer >= L);

	$effect(() => {
		if (!playing || !sw) return;
		const timer = setInterval(() => {
			if (layer >= L) playing = false;
			else layer += 1;
		}, 600);
		return () => clearInterval(timer);
	});

	function toggle() {
		if (done) layer = 0;
		playing = done || !playing;
	}

	// Each layer is drawn as a few representative states; the label gives the real number.
	const DOTS = 6;
	const geom = $derived.by(() => {
		const cols = L + 1;
		const left = 18;
		const gap = Math.max(26, Math.min(56, (width - left * 2) / cols));
		const top = 14;
		const dy = 26;
		const x = (c: number) => left + c * gap + 6;
		const y = (r: number) => top + r * dy;
		return { cols, gap, x, y, h: top + DOTS * dy + 34, w: left * 2 + cols * gap };
	});
	const edges = $derived.by(() => {
		const out: { c: number; d: string }[] = [];
		for (let c = 1; c <= L; c++) {
			const fromRows = c === 1 ? [Math.floor(DOTS / 2)] : [...Array(DOTS).keys()];
			for (const r0 of fromRows)
				for (let r1 = 0; r1 < DOTS; r1++)
					out.push({ c, d: `M${geom.x(c - 1)},${geom.y(r0)}L${geom.x(c)},${geom.y(r1)}` });
		}
		return out;
	});
</script>

<div class="sweep" bind:clientWidth={width}>
	{#if sw}
		<div class="controls">
			<button class="primary" onclick={toggle}>{playing ? 'Pause' : done ? 'Sweep again' : layer ? 'Resume' : 'Sweep'}</button>
			<button onclick={() => ((playing = false), (layer = Math.min(L, layer + 1)))}>Step ▶</button>
			<button onclick={() => ((playing = false), (layer = 0))}>Reset</button>
		</div>
		<svg width={geom.w} height={geom.h} role="img" aria-label="A layered graph of sequences of length 13, one column per position, swept from left to right">
			{#each edges as e}
				<path d={e.d} class="edge" class:lit={e.c <= layer} class:front={e.c === layer} />
			{/each}
			{#each Array(geom.cols) as _, c}
				{@const rows = c === 0 ? [Math.floor(DOTS / 2)] : [...Array(DOTS).keys()]}
				{#each rows as r}
					<circle cx={geom.x(c)} cy={geom.y(r)} r="5" class="node" class:lit={c <= layer} class:front={c === layer} />
				{/each}
				<text x={geom.x(c)} y={geom.y(DOTS - 1) + 22} text-anchor="middle" class="lab">{c === 0 ? 'start' : c}</text>
			{/each}
		</svg>
		<div class="readout mono" aria-live="polite">
			{#if layer === 0}
				<span class="muted">Each column is one position of a length-{L} sequence; each dot stands for many states (the
					last few residues written). Press Sweep.</span>
			{:else if layer < L}
				<span>after position {layer}: every one of the {sci(20 ** layer, 2)} possible first {layer} residue{layer > 1 ? 's' : ''} summed,
					held in {fmt(sw.nodes_by_layer[layer])} states</span>
				<span>running total, ln W = {sw.log_mass[layer - 1].toFixed(3)}</span>
			{:else}
				<span>all {sci(20 ** L, 2)} sequences of length {L} summed in one sweep</span>
				<span>ln Z({L}) = <strong>{sw.log_mass[L - 1].toFixed(4)}</strong>, which is log_partition({L}) =
					{sw.log_partition.toFixed(4)}</span>
				<span>the real graph: {fmt(sw.n_nodes)} states and {fmt(sw.n_edges)} edges, each visited once</span>
			{/if}
		</div>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.sweep {
		display: grid;
		gap: 0.55rem;
		font-size: var(--sl-text-sm);
	}
	.controls {
		display: flex;
		gap: 0.5rem;
	}
	button {
		font: inherit;
		font-size: var(--sl-text-sm);
		padding: 0.2rem 0.75rem;
		border-radius: 999px;
		border: 1px solid var(--sl-color-gray-5);
		background: transparent;
		color: var(--sl-color-white);
		cursor: pointer;
	}
	button.primary {
		background: var(--sl-color-accent);
		border-color: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	svg {
		max-width: 100%;
	}
	.edge {
		stroke: var(--sl-color-gray-5);
		stroke-width: 0.6;
		opacity: 0.35;
		transition: stroke 0.3s ease, opacity 0.3s ease;
	}
	.edge.lit {
		stroke: var(--sl-color-accent);
		opacity: 0.35;
	}
	.edge.front {
		stroke: var(--sl-color-text-accent);
		opacity: 0.9;
		stroke-width: 1;
	}
	.node {
		fill: var(--sl-color-gray-5);
		transition: fill 0.3s ease;
	}
	.node.lit {
		fill: var(--sl-color-accent);
	}
	.node.front {
		fill: var(--sl-color-white);
	}
	.lab {
		font-size: 10px;
		fill: var(--sl-color-gray-3);
	}
	.readout {
		display: grid;
		gap: 0.2rem;
		font-size: var(--sl-text-xs);
		padding: 0.5rem 0.7rem;
		border-left: 3px solid var(--sl-color-accent);
		background: var(--sl-color-gray-7, var(--sl-color-black));
		min-height: 3.6rem;
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
