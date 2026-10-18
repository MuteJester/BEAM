<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, onThemeChange, themeColors } from './theme';

	interface MapData {
		labels: string[];
		names: string[];
		distance: number[][];
		clusters: number[];
		linkage: number[][];
		order: number[];
		cut: number;
	}

	let data = $state<MapData | null>(null);
	let selected = $state<number | null>(null);
	let themeTick = $state(0);
	let failed = $state(false);
	let width = $state(560);

	$effect(() => {
		fetch(dataUrl('groups/cluster/map.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: MapData) => (data = d))
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	const groupColour: Record<string, string> = { A: '#438f91', B: '#ef785f', C: '#6c7bd9' };

	// Layout: the tree on top, the matrix below, one column and row per repertoire in tree order.
	const layout = $derived.by(() => {
		if (!data) return null;
		const n = data.names.length;
		const left = 58;
		const cell = Math.max(16, Math.min(32, (width - left - 16) / n));
		const treeH = 110;
		const band = 8;
		const top = treeH + band + 6;
		const pos = new Map<number, number>(); // leaf -> column
		data.order.forEach((leaf, k) => pos.set(leaf, k));
		// Each merge joins two clusters at a height; leaves are ids < n.
		const node: { x: number; h: number }[] = [];
		const xOf = (id: number) => (id < n ? left + (pos.get(id)! + 0.5) * cell : node[id - n].x);
		const hOf = (id: number) => (id < n ? 0 : node[id - n].h);
		const maxH = Math.max(...data.linkage.map((r) => r[2]));
		const y = (h: number) => treeH - (h / maxH) * (treeH - 10);
		const links: string[] = [];
		data.linkage.forEach(([a, b, h]) => {
			const xa = xOf(a);
			const xb = xOf(b);
			links.push(`M${xa},${y(hOf(a))}V${y(h)}H${xb}V${y(hOf(b))}`);
			node.push({ x: (xa + xb) / 2, h });
		});
		const values = data.distance.flat().filter((d) => d > 0);
		const colour = Plot.scale({
			color: { type: 'linear', scheme: 'viridis', domain: [0, Math.max(...values)] },
		});
		return { n, left, cell, treeH, band, top, links, cutY: y(data.cut), colour };
	});

	const info = $derived.by(() => {
		if (!data || selected === null) return null;
		const i = selected;
		const row = data.distance[i];
		let nearest = -1;
		row.forEach((d, j) => {
			if (j !== i && (nearest < 0 || d < row[nearest])) nearest = j;
		});
		const means: Record<string, number> = {};
		for (const g of ['A', 'B', 'C']) {
			const ds = row.filter((_, j) => j !== i && data!.labels[j] === g);
			means[g] = ds.reduce((s, d) => s + d, 0) / ds.length;
		}
		return { i, nearest, means };
	});
</script>

<div class="map" bind:clientWidth={width}>
	{#if data && layout}
		{@const L = layout}
		<svg
			width={L.left + L.n * L.cell + 96}
			height={L.top + L.n * L.cell + 54}
			role="group"
			aria-label="Clustering tree of twelve repertoires above the matrix of their distances; colour bands give each repertoire's group"
		>
			<!-- the tree, from the distances alone -->
			{#each L.links as d}
				<path {d} class="link" />
			{/each}
			<line x1={L.left} x2={L.left + L.n * L.cell + 6} y1={L.cutY} y2={L.cutY} class="cut" />
			<text x={L.left + L.n * L.cell + 10} y={L.cutY - 2} class="cut-label">cut into</text>
			<text x={L.left + L.n * L.cell + 10} y={L.cutY + 11} class="cut-label">3 clusters</text>

			<!-- the group each repertoire came from -->
			{#each data.order as leaf, k}
				<rect x={L.left + k * L.cell + 1} y={L.treeH + 2} width={L.cell - 2} height={L.band} fill={groupColour[data.labels[leaf]]} />
				<rect x={L.left - L.band - 6} y={L.top + k * L.cell + 1} width={L.band} height={L.cell - 2} fill={groupColour[data.labels[leaf]]} />
			{/each}

			<!-- the distance matrix, in tree order -->
			{#each data.order as ri, r}
				{#each data.order as ci, c}
					<rect
						x={L.left + c * L.cell}
						y={L.top + r * L.cell}
						width={L.cell - 1}
						height={L.cell - 1}
						fill={L.colour.apply(data.distance[ri][ci])}
						class:hl={selected !== null && (ri === selected || ci === selected)}
						class:dim={selected !== null && ri !== selected && ci !== selected}
					><title>{data.names[ri]} – {data.names[ci]}: {data.distance[ri][ci].toFixed(4)}</title></rect>
				{/each}
				<text
					x={L.left - L.band - 10}
					y={L.top + r * L.cell + L.cell / 2 + 4}
					text-anchor="end"
					class="name"
					class:sel={selected === ri}
					role="button"
					tabindex="0"
					aria-pressed={selected === ri}
					onclick={() => (selected = selected === ri ? null : ri)}
					onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (selected = selected === ri ? null : ri)}
				>{data.names[ri]}</text>
				<text x={L.left + r * L.cell + L.cell / 2} y={L.top + L.n * L.cell + 14} text-anchor="middle" class="name small">{data.names[ri]}</text>
			{/each}
		</svg>
		<div class="legend">
			{#each ['A', 'B', 'C'] as g}
				<span><span class="swatch" style:background={groupColour[g]}></span>group {g}</span>
			{/each}
			<span class="muted">matrix colour: distance, dark = close</span>
		</div>
		<p class="readout" aria-live="polite">
			{#if info}
				<strong>{data.names[info.i]}</strong> (group {data.labels[info.i]}): nearest repertoire
				<strong>{data.names[info.nearest]}</strong> (group {data.labels[info.nearest]}) at
				{data.distance[info.i][info.nearest].toFixed(3)}. Mean distance to the other repertoires of each group:
				{#each ['A', 'B', 'C'] as g, k}{k ? ', ' : ''}{g} {info.means[g].toFixed(3)}{/each}.
			{:else}
				Choose a repertoire's name to see its nearest neighbour and its distances to each group.
			{/if}
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
		gap: 0.5rem;
		font-size: var(--sl-text-sm);
	}
	svg {
		max-width: 100%;
		overflow: visible;
	}
	.link {
		fill: none;
		stroke: var(--sl-color-gray-3);
		stroke-width: 1.5;
	}
	.cut {
		stroke: #ef785f;
		stroke-dasharray: 4 3;
	}
	.cut-label {
		font-size: 11px;
		fill: #ef785f;
	}
	.name {
		font-size: 11px;
		fill: var(--sl-color-gray-2);
		cursor: pointer;
	}
	.name.small {
		font-size: 10px;
		cursor: default;
	}
	.name.sel {
		fill: var(--sl-color-white);
		font-weight: 700;
	}
	rect.dim {
		opacity: 0.45;
	}
	rect.hl {
		stroke: var(--sl-color-white);
		stroke-width: 0.6;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem 1rem;
		font-size: var(--sl-text-xs);
		color: var(--sl-color-gray-2);
	}
	.swatch {
		display: inline-block;
		width: 0.75rem;
		height: 0.75rem;
		border-radius: 2px;
		margin-inline-end: 0.3rem;
		vertical-align: -0.1rem;
	}
	.readout {
		margin: 0;
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
