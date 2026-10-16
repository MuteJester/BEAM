<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, onThemeChange, themeColors } from './theme';

	interface MapData {
		labels: string[];
		names: string[];
		distance: number[][];
		xy: number[][];
		clusters: number[];
	}

	let data = $state<MapData | null>(null);
	let selected = $state(0);
	let themeTick = $state(0);
	let failed = $state(false);
	let heat: HTMLDivElement | undefined = $state();
	let scatter: HTMLDivElement | undefined = $state();

	$effect(() => {
		fetch(dataUrl('groups/cluster/map.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: MapData) => (data = d))
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	const groupColour = (g: string, c: ReturnType<typeof themeColors>) =>
		({ A: c.accent, B: c.second, C: '#6c7bd9' })[g] ?? c.muted;

	const nearest = $derived.by(() => {
		if (!data) return -1;
		const row = data.distance[selected];
		let best = -1;
		row.forEach((d, j) => {
			if (j !== selected && (best < 0 || d < row[best])) best = j;
		});
		return best;
	});

	$effect(() => {
		themeTick;
		if (!data || !heat || !scatter) return;
		const c = themeColors();
		const cells = data.distance.flatMap((row, i) =>
			row.map((d, j) => ({ i: data!.names[i], j: data!.names[j], d })),
		);
		const h = Plot.plot({
			width: Math.min(heat.clientWidth, 380),
			height: 340,
			marginLeft: 36,
			marginBottom: 36,
			style: { color: c.text, background: 'transparent', fontSize: '11px' },
			x: { domain: data.names, label: null, tickRotate: -90 },
			y: { domain: data.names, label: null },
			color: { scheme: 'viridis', legend: true, label: 'distance' },
			marks: [
				Plot.cell(cells, { x: 'j', y: 'i', fill: 'd', inset: 0.5 }),
				Plot.cell(
					cells.filter((r) => r.i === data!.names[selected] || r.j === data!.names[selected]),
					{ x: 'j', y: 'i', stroke: c.text, fill: 'none', strokeWidth: 0.6 },
				),
			],
		});
		heat.replaceChildren(h);
		const pts = data.xy.map(([x, y], k) => ({ x, y, name: data!.names[k], g: data!.labels[k], k }));
		const s = Plot.plot({
			width: Math.min(scatter.clientWidth, 380),
			height: 340,
			inset: 24,
			style: { color: c.text, background: 'transparent', fontSize: '11px' },
			x: { axis: null },
			y: { axis: null },
			marks: [
				Plot.link([{ a: pts[selected], b: pts[nearest] }], {
					x1: (d) => d.a.x,
					y1: (d) => d.a.y,
					x2: (d) => d.b.x,
					y2: (d) => d.b.y,
					stroke: c.text,
					strokeDasharray: '3,3',
				}),
				Plot.dot(pts, {
					x: 'x',
					y: 'y',
					r: (d) => (d.k === selected ? 9 : 6),
					fill: (d) => groupColour(d.g, c),
					stroke: (d) => (d.k === selected ? c.text : 'none'),
				}),
				// Within-group differences are spread over many dimensions and nearly
				// vanish in two, so members of a group overlap: label only the selected
				// repertoire and its nearest neighbour.
				Plot.text(
					pts.filter((d) => d.k === selected || d.k === nearest),
					{ x: 'x', y: 'y', text: 'name', dy: -14, fill: c.text, fontWeight: 600 },
				),
			],
		});
		scatter.replaceChildren(s);
	});
</script>

<div class="map">
	{#if data}
		<div class="row" role="group" aria-label="Repertoire">
			<span class="muted">Repertoire</span>
			{#each data.names as name, k}
				<button class:active={selected === k} aria-pressed={selected === k} onclick={() => (selected = k)}>{name}</button>
			{/each}
		</div>
		<div class="panels">
			<div bind:this={heat} class="panel" role="img" aria-label="Distance between every pair of repertoires"></div>
			<div bind:this={scatter} class="panel" role="img" aria-label="The repertoires laid out so that distances are preserved as well as possible"></div>
		</div>
		<p class="readout">
			<strong>{data.names[selected]}</strong> (group {data.labels[selected]}): nearest repertoire
			<strong>{data.names[nearest]}</strong> (group {data.labels[nearest]}), at distance
			{data.distance[selected][nearest].toFixed(4)}.
		</p>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.map {
		display: grid;
		gap: 0.6rem;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		align-items: center;
	}
	.panels {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
	}
	.panel {
		flex: 1 1 300px;
		min-height: 340px;
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
	.readout {
		font-size: var(--sl-text-sm);
		margin: 0;
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
