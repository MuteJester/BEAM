<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, fmt, onThemeChange, themeColors } from './theme';

	interface Data {
		sequences: string[];
		log_p: number[];
		length_distribution: Record<string, number>;
	}

	let data = $state<Data | null>(null);
	let drawn = $state(0);
	let playing = $state(false);
	let speed = $state(20); // draws per tick
	let themeTick = $state(0);
	let chart: HTMLDivElement | undefined = $state();

	$effect(() => {
		fetch(dataUrl('viz/sampling/sampling.json'))
			.then((r) => r.json())
			.then((d: Data) => (data = d));
		return onThemeChange(() => themeTick++);
	});

	$effect(() => {
		if (!playing || !data) return;
		const timer = setInterval(() => {
			const next = Math.min(data!.sequences.length, drawn + (drawn < 30 ? 1 : speed));
			drawn = next;
			if (next >= data!.sequences.length) playing = false;
		}, 120);
		return () => clearInterval(timer);
	});

	const recent = $derived(data ? data.sequences.slice(Math.max(0, drawn - 8), drawn).reverse() : []);
	const recentLogP = $derived(data ? data.log_p.slice(Math.max(0, drawn - 8), drawn).reverse() : []);
	const counts = $derived.by(() => {
		const c = new Map<number, number>();
		if (data) for (const s of data.sequences.slice(0, drawn)) c.set(s.length, (c.get(s.length) ?? 0) + 1);
		return c;
	});

	$effect(() => {
		themeTick;
		if (!data || !chart) return;
		const c = themeColors();
		const lengths = Object.keys(data.length_distribution)
			.map(Number)
			.filter((L) => L >= 6 && L <= 24);
		const observed = lengths.map((L) => ({ L, share: drawn ? (counts.get(L) ?? 0) / drawn : 0 }));
		const exact = lengths.map((L) => ({ L, share: data!.length_distribution[L] }));
		const plot = Plot.plot({
			width: chart.clientWidth,
			height: 240,
			marginLeft: 48,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { label: 'CDR3 length (residues) →', tickFormat: 'd' },
			y: { label: '↑ Share', grid: true, domain: [0, Math.max(0.26, ...observed.map((d) => d.share))] },
			marks: [
				Plot.rectY(observed, { x1: (d) => d.L - 0.4, x2: (d) => d.L + 0.4, y: 'share', fill: c.accent }),
				Plot.dot(exact, { x: 'L', y: 'share', fill: c.second, r: 3.5 }),
				Plot.ruleY([0], { stroke: c.muted }),
			],
		});
		chart.replaceChildren(plot);
	});
</script>

<div class="sampling">
	{#if data}
		<div class="controls">
			<button onclick={() => (playing = !playing)}>
				{playing ? 'Pause' : drawn >= data.sequences.length ? 'Done' : drawn ? 'Resume' : 'Draw'}
			</button>
			<button
				onclick={() => {
					playing = false;
					drawn = 0;
				}}>Reset</button
			>
			<label class="speed">
				Speed
				<input type="range" min="1" max="100" bind:value={speed} aria-label="Draws per step" />
			</label>
			<span class="count"><strong>{fmt(drawn)}</strong> of {fmt(data.sequences.length)} drawn</span>
		</div>
		<ol class="ticker" aria-live="off">
			{#each recent as s, i (drawn - i)}
				<li><span class="mono">{s}</span><span class="logp">ln P = {recentLogP[i].toFixed(1)}</span></li>
			{/each}
		</ol>
		<div bind:this={chart} class="chart"></div>
		<div class="legend">
			<span><span class="swatch a"></span>share of the draws so far</span>
			<span><span class="swatch b"></span>the model's exact length distribution</span>
		</div>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.sampling {
		display: grid;
		gap: 0.6rem;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		align-items: center;
		font-size: var(--sl-text-sm);
	}
	button {
		flex: none;
		font: inherit;
		font-size: var(--sl-text-xs);
		padding: 0.2rem 0.7rem;
		border-radius: 999px;
		border: 1px solid var(--sl-color-gray-5);
		background: transparent;
		color: var(--sl-color-white);
		cursor: pointer;
	}
	.speed {
		display: flex;
		gap: 0.4rem;
		align-items: center;
		color: var(--sl-color-gray-3);
	}
	input[type='range'] {
		accent-color: var(--sl-color-accent);
	}
	.count {
		margin-left: auto;
	}
	.ticker {
		list-style: none;
		margin: 0;
		padding: 0;
		height: 10.5rem;
		overflow: hidden;
		font-size: var(--sl-text-sm);
	}
	.ticker li {
		display: flex;
		justify-content: space-between;
		padding: 0.1rem 0;
		border-bottom: 1px solid var(--sl-color-gray-6);
	}
	.ticker li:first-child {
		color: var(--sl-color-accent-high);
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.logp {
		color: var(--sl-color-gray-3);
	}
	.chart {
		min-height: 240px;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		font-size: var(--sl-text-xs);
		color: var(--sl-color-gray-3);
	}
	.swatch {
		display: inline-block;
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 2px;
		margin-right: 0.35rem;
	}
	.swatch.a {
		background: var(--sl-color-accent);
	}
	.swatch.b {
		background: #ef785f;
		border-radius: 50%;
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
