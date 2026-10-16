<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { boundUp, dataUrl, fmt, onThemeChange, pct, themeColors } from './theme';

	interface Grid {
		depths: number[];
		richness: number[];
		richness_bound: number[];
		sharing: number[];
		sharing_bound: number[];
		novelty: number[];
		entropy: number;
		sample_entropy: number[];
	}

	let data = $state<Grid | null>(null);
	let index = $state(40); // 10^(3 + 40/20) = 100,000 cells
	let failed = $state(false);
	let themeTick = $state(0);
	let chart: HTMLDivElement | undefined = $state();

	$effect(() => {
		fetch(dataUrl('viz/depth_grid/depth_grid.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: Grid) => (data = d))
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	const depth = $derived(data ? data.depths[index] : 0);
	const distinct = $derived(data ? data.richness[index] : 0);
	const shared = $derived(data ? data.sharing[index] : 0);
	const bound = $derived(data ? Math.max(data.richness_bound[index], data.sharing_bound[index]) : 0);

	$effect(() => {
		themeTick;
		if (!data || !chart) return;
		const c = themeColors();
		const rows = data.depths.map((d, k) => ({ d, r: data!.richness[k], s: data!.sharing[k] }));
		const plot = Plot.plot({
			width: chart.clientWidth,
			height: 300,
			marginLeft: 64,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { type: 'log', label: 'Cells sampled (depth) →', grid: true },
			y: { type: 'log', label: '↑ Sequences', grid: true },
			marks: [
				Plot.line(rows, { x: 'd', y: 'd', stroke: c.muted, strokeDasharray: '4,4' }),
				Plot.line(rows, { x: 'd', y: 'r', stroke: c.accent, strokeWidth: 2.5 }),
				Plot.line(rows, { x: 'd', y: 's', stroke: c.second, strokeWidth: 2 }),
				Plot.ruleX([depth], { stroke: c.muted }),
				Plot.dot([{ d: depth, v: distinct }], { x: 'd', y: 'v', fill: c.accent, r: 5 }),
				Plot.dot([{ d: depth, v: shared }], { x: 'd', y: 'v', fill: c.second, r: 5 }),
				Plot.text([{ d: 2e3, v: 2e3 * 1.6 }], {
					x: 'd',
					y: 'v',
					text: () => 'every cell a new sequence',
					fill: c.muted,
					textAnchor: 'start',
					rotate: -34,
				}),
			],
		});
		chart.replaceChildren(plot);
	});
</script>

<div class="depth-slider">
	{#if data}
		<label class="control">
			<span>Depth: <strong>{fmt(depth)}</strong> cells</span>
			<input
				type="range"
				min="0"
				max={data.depths.length - 1}
				step="1"
				bind:value={index}
				aria-label="Number of cells sampled"
			/>
		</label>
		<div class="readout">
			<div>
				<span class="swatch accent"></span>
				<strong>{fmt(distinct)}</strong> distinct sequences expected
				<span class="muted">(that is {pct(distinct / depth)} of the number of cells sampled)</span>
			</div>
			<div>
				<span class="swatch second"></span>
				<strong>{fmt(shared)}</strong> of them also expected in a second sample of the same depth
				<span class="muted">({pct(shared / distinct)})</span>
			</div>
			<div>
				The next cell has a <strong>{pct(data.novelty[index])}</strong> chance of carrying a sequence not yet in the
				sample; the sample's Shannon entropy is expected to reach <strong>{pct(data.sample_entropy[index] / data.entropy)}</strong>
				of the population's.
			</div>
			<div class="muted">Exact to a relative error of at most {boundUp(bound)}.</div>
		</div>
		<div bind:this={chart} class="chart"></div>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading the exact values…</p>
	{/if}
</div>

<style>
	.depth-slider {
		display: grid;
		gap: 0.75rem;
	}
	.control {
		display: grid;
		gap: 0.35rem;
	}
	input[type='range'] {
		width: 100%;
		accent-color: var(--sl-color-accent);
	}
	.readout {
		display: grid;
		gap: 0.25rem;
		font-size: var(--sl-text-sm);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
	.swatch {
		display: inline-block;
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 50%;
		margin-right: 0.35rem;
		vertical-align: baseline;
	}
	.swatch.accent {
		background: var(--sl-color-accent);
	}
	.swatch.second {
		background: #ef785f;
	}
	.chart {
		min-height: 300px;
	}
</style>
