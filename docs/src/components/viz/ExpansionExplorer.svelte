<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, onThemeChange, pct, sci, themeColors } from './theme';

	interface Row {
		strength: number;
		top_clone: string;
		top_share: number;
		top10_share: number;
		p_top_weighted: number;
		p_top_unique: number;
		distance: number;
		length_weighted: Record<string, number>;
		length_unique: Record<string, number>;
	}

	let rows = $state<Row[] | null>(null);
	let index = $state(4);
	let themeTick = $state(0);
	let failed = $state(false);
	let chart: HTMLDivElement | undefined = $state();

	$effect(() => {
		fetch(dataUrl('viz/expansion/expansion.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d) => (rows = d.rows))
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	const row = $derived(rows?.[index]);

	$effect(() => {
		themeTick;
		if (!row || !chart) return;
		const c = themeColors();
		const lines = [
			...Object.entries(row.length_weighted).map(([L, p]) => ({ L: +L, p, model: 'weighted by cells' })),
			...Object.entries(row.length_unique).map(([L, p]) => ({ L: +L, p, model: 'each sequence once' })),
		];
		const plot = Plot.plot({
			width: chart.clientWidth,
			height: 230,
			marginLeft: 48,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { label: 'CDR3 length →', tickFormat: 'd' },
			y: { label: '↑ Probability of the length', grid: true },
			color: { domain: ['weighted by cells', 'each sequence once'], range: [c.second, c.accent], legend: true },
			marks: [Plot.line(lines, { x: 'L', y: 'p', stroke: 'model', strokeWidth: 2, marker: 'circle' })],
		});
		chart.replaceChildren(plot);
	});
</script>

<div class="expansion">
	{#if rows && row}
		<label class="control">
			<span>Clonal expansion: <strong>{row.strength === 0 ? 'none' : `strength ${row.strength}`}</strong></span>
			<input type="range" min="0" max={rows.length - 1} step="1" bind:value={index} aria-label="Strength of clonal expansion" />
		</label>
		<div class="readout" aria-live="polite">
			<div>Largest clone: <strong>{pct(row.top_share)}</strong> of the cells; ten largest: {pct(row.top10_share)}.</div>
			<div>
				Its probability: <strong>{sci(row.p_top_weighted)}</strong> weighted by cells, <strong>{sci(row.p_top_unique)}</strong>
				with each sequence counted once.
			</div>
			<div>Distance between the two models: <strong>{row.distance.toFixed(4)}</strong>.</div>
		</div>
		<div bind:this={chart} role="img" aria-label="CDR3 length distribution of the repertoire weighted by cells and with each sequence counted once"></div>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.expansion {
		display: grid;
		gap: 0.6rem;
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
		gap: 0.2rem;
		font-size: var(--sl-text-sm);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
