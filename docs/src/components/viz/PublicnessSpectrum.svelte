<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, fmt, onThemeChange, pct, sci, themeColors } from './theme';

	type Spectra = Record<string, { n: number; depth: number; counts: number[] }>;

	let data = $state<Spectra | null>(null);
	let size = $state(100);
	let depth = $state(100_000);
	let themeTick = $state(0);
	let failed = $state(false);
	let chart: HTMLDivElement | undefined = $state();
	const sizes = [10, 100, 1000];
	const depths = [10_000, 100_000, 1_000_000];

	$effect(() => {
		fetch(dataUrl('viz/publicness_spectrum/publicness_spectrum.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: Spectra) => (data = d))
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	const counts = $derived(data?.[`${size}x${depth}`]?.counts ?? []);

	// As in the paper's spectrum panel: sequences per unit publicness on
	// log-spaced bins, drawn at each bin's geometric centre, with the top bin
	// dropped. Sequences common enough to be in nearly everyone pile up there,
	// at "carried by all", which is a boundary, not part of the curve.
	function binned(c: number[], perDecade = 8): { k: number; v: number }[] {
		const n = c.length;
		const steps = Math.floor(perDecade * Math.log10(n + 1)) + 1;
		const edges = new Set<number>([1, n + 1]);
		for (let i = 0; i <= steps; i++) edges.add(Math.round(10 ** ((i / steps) * Math.log10(n + 1))));
		const e = [...edges].filter((x) => x >= 1 && x <= n + 1).sort((a, b) => a - b);
		const out: { k: number; v: number }[] = [];
		for (let i = 0; i + 1 < e.length; i++) {
			const lo = e[i];
			const hi = e[i + 1];
			let sum = 0;
			let logs = 0;
			for (let k = lo; k < hi; k++) {
				sum += c[k - 1];
				logs += Math.log(k);
			}
			out.push({ k: Math.exp(logs / (hi - lo)), v: sum / (hi - lo) });
		}
		return out.slice(0, -1);
	}
	const seen = $derived(counts.reduce((a, b) => a + b, 0));
	const everyone = $derived(counts.length ? counts[counts.length - 1] : 0);

	$effect(() => {
		themeTick;
		if (!counts.length || !chart) return;
		const c = themeColors();
		const pts = binned(counts).filter((d) => d.v >= 0.01);
		const plot = Plot.plot({
			width: chart.clientWidth,
			height: 280,
			marginLeft: 64,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { type: 'log', label: `Number of people carrying a sequence (out of ${size}) →` },
			y: { type: 'log', label: '↑ Expected sequences per unit publicness', grid: true },
			marks: [
				Plot.line(pts, { x: 'k', y: 'v', stroke: c.accent, strokeWidth: 1.5 }),
				Plot.dot(pts, { x: 'k', y: 'v', fill: c.accent, r: 3 }),
			],
		});
		chart.replaceChildren(plot);
	});
</script>

<div class="spectrum">
	{#if data}
		<div class="row" role="group" aria-label="Cohort size">
			<span class="muted">Cohort</span>
			{#each sizes as s}
				<button class:active={size === s} aria-pressed={size === s} onclick={() => (size = s)}>{fmt(s)} people</button>
			{/each}
		</div>
		<div class="row" role="group" aria-label="Cells sampled per person">
			<span class="muted">Depth</span>
			{#each depths as d}
				<button class:active={depth === d} aria-pressed={depth === d} onclick={() => (depth = d)}>{fmt(d)} cells each</button>
			{/each}
		</div>
		<div class="readout" aria-live="polite">
			<div><strong>{fmt(seen)}</strong> distinct sequences expected across the cohort.</div>
			<div><strong>{pct(counts[0] / seen)}</strong> of them private (in one person); {everyone >= 0.1 ? fmt(everyone, 1) : sci(everyone)} expected in all {fmt(size)}.</div>
		</div>
		<div bind:this={chart} role="img" aria-label="Expected number of sequences carried by each number of people in the cohort"></div>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.spectrum {
		display: grid;
		gap: 0.6rem;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: center;
	}
	.row .muted {
		min-width: 4.5rem;
		font-size: var(--sl-text-sm);
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
	.readout {
		font-size: var(--sl-text-sm);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
