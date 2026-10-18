<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, fmt, onThemeChange, pct, sci, themeColors } from './theme';

	interface Pmf {
		k0: number;
		p: number[];
	}
	interface Cohort {
		n: number;
		depth: number;
		grid: Pmf[];
		examples: Record<string, Pmf>;
		means: number[];
	}
	interface Data {
		probabilities: number[];
		examples: Record<string, number>;
		cohorts: Record<string, Cohort>;
	}

	const sizes = [10, 100, 500];
	const depths = [10_000, 100_000, 1_000_000];

	let data = $state<Data | null>(null);
	let size = $state(100);
	let failed = $state(false);
	let depth = $state(100_000);
	let sequence = $state<string | null>('CASSLGQGAYNEQFF');
	let gridIndex = $state(50);
	let themeTick = $state(0);
	let chart: HTMLDivElement | undefined = $state();

	$effect(() => {
		fetch(dataUrl('viz/publicness_grid/publicness_grid.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: Data) => (data = d))
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	const cohort = $derived(data?.cohorts[`${size}x${depth}`]);
	const exampleNames = $derived(data ? Object.keys(data.examples) : []);
	const probability = $derived(
		data ? (sequence ? data.examples[sequence] : data.probabilities[gridIndex]) : 0,
	);
	const pmf = $derived(cohort ? (sequence ? cohort.examples[sequence] : cohort.grid[gridIndex]) : null);
	const mean = $derived(
		cohort && data
			? sequence
				? cohort.means[data.probabilities.length + exampleNames.indexOf(sequence)]
				: cohort.means[gridIndex]
			: 0,
	);
	const rows = $derived(pmf ? pmf.p.map((p, j) => ({ k: pmf.k0 + j, p })) : []);
	const pNobody = $derived(pmf && pmf.k0 === 0 ? pmf.p[0] : 0);
	const mode = $derived(rows.length ? rows.reduce((a, b) => (b.p > a.p ? b : a)).k : 0);

	$effect(() => {
		themeTick;
		if (!chart || !rows.length) return;
		const c = themeColors();
		const plot = Plot.plot({
			width: chart.clientWidth,
			height: 260,
			marginLeft: 52,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { label: `People carrying it (out of ${size}) →` },
			y: { label: '↑ Probability', grid: true },
			marks: [
				Plot.rectY(rows, { x1: (d) => d.k - 0.45, x2: (d) => d.k + 0.45, y: 'p', fill: c.accent }),
				Plot.ruleY([0], { stroke: c.muted }),
			],
		});
		chart.replaceChildren(plot);
	});
</script>

<div class="explorer">
	{#if data && cohort}
		<div class="row" role="group" aria-label="Cohort size">
			<span class="label">Cohort</span>
			{#each sizes as s}
				<button class:active={size === s} aria-pressed={size === s} onclick={() => (size = s)}>{s} people</button>
			{/each}
		</div>
		<div class="row" role="group" aria-label="Cells sampled per person">
			<span class="label">Depth</span>
			{#each depths as d}
				<button class:active={depth === d} aria-pressed={depth === d} onclick={() => (depth = d)}>{fmt(d)} cells each</button>
			{/each}
		</div>
		<div class="row" role="group" aria-label="Sequence">
			<span class="label">Sequence</span>
			{#each exampleNames as s}
				<button class="mono" class:active={sequence === s} aria-pressed={sequence === s} onclick={() => (sequence = s)}>{s}</button>
			{/each}
		</div>
		<label class="control">
			<span>…or any probability: <strong>{sci(probability)}</strong></span>
			<input
				type="range"
				min="0"
				max={data.probabilities.length - 1}
				step="1"
				value={sequence ? data.probabilities.findIndex((p) => p >= probability) : gridIndex}
				oninput={(e) => {
					gridIndex = Number((e.target as HTMLInputElement).value);
					sequence = null;
				}}
				aria-label="Sequence probability"
			/>
		</label>
		<div class="readout" aria-live="polite">
			<div><strong>{fmt(mean, 1)}</strong> people expected to carry it</div>
			<div class="muted">
				Nobody: {pct(pNobody)} · most likely: {mode} {mode === 1 ? 'person' : 'people'}
			</div>
		</div>
		<div bind:this={chart} class="chart" role="img" aria-label="Probability of each number of carriers in the cohort"></div>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.explorer {
		display: grid;
		gap: 0.6rem;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: center;
	}
	.label {
		min-width: 5.5rem;
		font-size: var(--sl-text-sm);
		color: var(--sl-color-gray-3);
	}
	button {
		flex: none;
		font: inherit;
		font-size: var(--sl-text-xs);
		padding: 0.2rem 0.6rem;
		border-radius: 999px;
		border: 1px solid var(--sl-color-gray-5);
		background: transparent;
		color: var(--sl-color-white);
		cursor: pointer;
	}
	button.active {
		background: var(--sl-color-accent);
		border-color: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.control {
		display: grid;
		gap: 0.35rem;
		font-size: var(--sl-text-sm);
	}
	input[type='range'] {
		width: 100%;
		accent-color: var(--sl-color-accent);
	}
	.readout {
		font-size: var(--sl-text-sm);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
	.chart {
		min-height: 260px;
	}
</style>
