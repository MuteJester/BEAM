<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, onThemeChange, pct, themeColors } from './theme';

	interface LengthRow {
		length: number;
		p: number;
		q: number;
		length_term: number;
		pattern_term: number;
	}
	interface Pattern {
		length: number;
		start: number;
		word: string;
		p: number;
		q: number;
		contribution: number;
	}
	interface Data {
		squared_distance: number;
		length_squared: number;
		pattern_squared: number;
		lengths: LengthRow[];
		patterns: Pattern[];
	}

	let data = $state<Data | null>(null);
	let selected = $state<number | null>(null);
	let themeTick = $state(0);
	let chart: HTMLDivElement | undefined = $state();

	$effect(() => {
		fetch(dataUrl('viz/distance_breakdown/distance_breakdown.json'))
			.then((r) => r.json())
			.then((d: Data) => {
				data = d;
				const top = [...d.lengths].sort((a, b) => b.pattern_term - a.pattern_term)[0];
				selected = top?.length ?? null;
			});
		return onThemeChange(() => themeTick++);
	});

	const shown = $derived(
		data ? data.lengths.filter((r) => r.length_term + r.pattern_term > data!.squared_distance * 0.002) : [],
	);
	const patterns = $derived(
		data && selected !== null ? data.patterns.filter((p) => p.length === selected).slice(0, 8) : [],
	);

	$effect(() => {
		themeTick;
		if (!data || !chart) return;
		const c = themeColors();
		const rows = shown.flatMap((r) => [
			{ length: r.length, part: 'length', v: r.length_term },
			{ length: r.length, part: 'local patterns', v: r.pattern_term },
		]);
		const plot = Plot.plot({
			width: chart.clientWidth,
			height: 240,
			marginLeft: 56,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { label: 'CDR3 length (residues) →', type: 'band', padding: 0.2 },
			y: { label: '↑ Contribution to the squared distance', grid: true },
			color: { domain: ['length', 'local patterns'], range: [c.second, c.accent] },
			marks: [
				Plot.barY(rows, {
					x: 'length',
					y: 'v',
					fill: 'part',
					stroke: (d) => (d.length === selected ? c.text : 'none'),
				}),
				Plot.ruleY([0], { stroke: c.muted }),
			],
		});
		plot.addEventListener('click', (e) => {
			const t = (e.target as Element).closest('rect');
			const datum = t && (t as any).__data__;
			if (typeof datum === 'number' && rows[datum]) selected = rows[datum].length;
		});
		chart.replaceChildren(plot);
	});
</script>

<div class="breakdown">
	{#if data}
		<div class="split" aria-label="Share of the squared distance">
			<div class="part length" style={`width: ${(100 * data.length_squared) / data.squared_distance}%`}></div>
			<div class="part patterns" style={`width: ${(100 * data.pattern_squared) / data.squared_distance}%`}></div>
		</div>
		<div class="legend">
			<span><span class="swatch length"></span>length: {pct(data.length_squared / data.squared_distance)}</span>
			<span><span class="swatch patterns"></span>local patterns: {pct(data.pattern_squared / data.squared_distance)}</span>
			<span class="muted">distance {Math.sqrt(data.squared_distance).toFixed(4)}</span>
		</div>
		<div bind:this={chart} class="chart"></div>
		<div class="row" role="group" aria-label="Length">
			<span class="muted">Patterns at length</span>
			{#each shown as r}
				<button class:active={selected === r.length} onclick={() => (selected = r.length)}>{r.length}</button>
			{/each}
		</div>
		{#if patterns.length}
			<table>
				<thead>
					<tr><th>positions</th><th>pattern</th><th>donor A</th><th>donor B</th><th>more common in</th></tr>
				</thead>
				<tbody>
					{#each patterns as p}
						<tr>
							<td>{p.start}–{p.start + p.word.length - 1}</td>
							<td class="mono">{p.word}</td>
							<td>{pct(p.p, 2)}</td>
							<td>{pct(p.q, 2)}</td>
							<td>{p.p > p.q ? 'A' : 'B'}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{:else}
			<p class="muted">No pattern among the strongest at this length.</p>
		{/if}
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.breakdown {
		display: grid;
		gap: 0.6rem;
	}
	.split {
		display: flex;
		height: 0.9rem;
		border-radius: 999px;
		overflow: hidden;
	}
	.part.length,
	.swatch.length {
		background: #ef785f;
	}
	.part.patterns,
	.swatch.patterns {
		background: var(--sl-color-accent);
	}
	.legend,
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.75rem;
		align-items: center;
		font-size: var(--sl-text-sm);
	}
	.row {
		gap: 0.3rem;
	}
	.swatch {
		display: inline-block;
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 2px;
		margin-right: 0.35rem;
	}
	button {
		flex: none;
		font: inherit;
		font-size: var(--sl-text-xs);
		padding: 0.1rem 0.5rem;
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
	table {
		font-size: var(--sl-text-sm);
		width: 100%;
		border-collapse: collapse;
	}
	th,
	td {
		text-align: left;
		padding: 0.2rem 0.5rem;
		border-bottom: 1px solid var(--sl-color-gray-6);
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
	.chart {
		min-height: 240px;
	}
</style>
