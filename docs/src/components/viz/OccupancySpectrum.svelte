<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, fmt, onThemeChange, pct, themeColors } from './theme';

	interface Row {
		depth: number;
		distinct: number;
		counts: number[];
		more: number;
		cells_more: number;
	}

	let data = $state<{ J: number; rows: Row[] } | null>(null);
	let index = $state(6);
	let view = $state<'sequences' | 'cells'>('sequences');
	let themeTick = $state(0);
	let failed = $state(false);
	let chart: HTMLDivElement | undefined = $state();

	$effect(() => {
		fetch(dataUrl('viz/occupancy_spectrum/occupancy_spectrum.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d) => {
				data = d;
				index = Math.min(index, d.rows.length - 1);
			})
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	const row = $derived(data?.rows[index]);
	// One bar per class: seen exactly j times, then "more than J".
	const bars = $derived.by(() => {
		if (!data || !row) return [];
		const out = row.counts.map((c, i) => ({
			label: String(i + 1),
			sequences: c,
			cells: ((i + 1) * c) / row.depth,
		}));
		out.push({ label: `>${data.J}`, sequences: row.more, cells: row.cells_more / row.depth });
		return out;
	});

	$effect(() => {
		themeTick;
		if (!chart || !bars.length) return;
		const c = themeColors();
		const cells = view === 'cells';
		const plot = Plot.plot({
			width: chart.clientWidth,
			height: 260,
			marginLeft: 64,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { type: 'band', domain: bars.map((b) => b.label), label: 'Times a sequence is seen in the sample →', padding: 0.15 },
			y: cells
				? { label: '↑ Share of the cells', grid: true, tickFormat: (v: number) => pct(v, 0) }
				: { type: 'log', label: '↑ Expected number of sequences', grid: true, tickFormat: (v: number) => fmt(v) },
			marks: [
				Plot.barY(
					bars.filter((b) => (cells ? b.cells > 0 : b.sequences >= 0.01)),
					{
						x: 'label',
						y: cells ? 'cells' : 'sequences',
						y1: cells ? 0 : 0.01,
						fill: (b) => (b.label.startsWith('>') ? c.second : c.accent),
					},
				),
			],
		});
		chart.replaceChildren(plot);
	});
</script>

<div class="occupancy">
	{#if data && row}
		<label class="depth">
			<span class="muted">Depth</span>
			<input
				type="range"
				min="0"
				max={data.rows.length - 1}
				step="1"
				bind:value={index}
				aria-label="Number of cells sampled, for the occupancy spectrum"
			/>
			<strong>{fmt(row.depth)} cells</strong>
		</label>
		<div class="row" role="group" aria-label="What the bars count">
			<button class:active={view === 'sequences'} aria-pressed={view === 'sequences'} onclick={() => (view = 'sequences')}>sequences</button>
			<button class:active={view === 'cells'} aria-pressed={view === 'cells'} onclick={() => (view = 'cells')}>share of cells</button>
		</div>
		<div bind:this={chart} role="img" aria-label="Expected number of sequences seen exactly 1 to 20 times, and more than 20 times, in a sample of the chosen depth"></div>
		<p class="readout" aria-live="polite">
			In {fmt(row.depth)} cells: <strong>{fmt(row.distinct)}</strong> distinct sequences, {pct(row.counts[0] / row.distinct)}
			of them seen once. {fmt(row.more)} are seen more than {data.J} times, and together they carry
			{pct(row.cells_more / row.depth)} of the cells.
		</p>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.occupancy {
		display: grid;
		gap: 0.6rem;
	}
	.depth {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		font-size: var(--sl-text-sm);
	}
	.depth input {
		flex: 1;
		accent-color: var(--sl-color-accent);
	}
	.row {
		display: flex;
		gap: 0.35rem;
	}
	button {
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
	.readout {
		font-size: var(--sl-text-sm);
		margin: 0;
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
