<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, fmt, onThemeChange, sci, themeColors } from './theme';

	interface Data {
		depth: number;
		computed: number;
		bound: number;
		sd: number;
		sd_bound: number;
		simulated: number[];
	}

	let data = $state<Data | null>(null);
	let shown = $state(0);
	let playing = $state(false);
	let themeTick = $state(0);
	let failed = $state(false);
	let chart: HTMLDivElement | undefined = $state();

	$effect(() => {
		fetch(dataUrl('viz/simulated/simulated.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: Data) => (data = d))
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	$effect(() => {
		if (!playing || !data) return;
		const timer = setInterval(() => {
			shown = Math.min(data!.simulated.length, shown + 1);
			if (shown >= data!.simulated.length) playing = false;
		}, 90);
		return () => clearInterval(timer);
	});

	const finished = $derived(data !== null && shown >= data.simulated.length);
	function toggle() {
		if (finished) shown = 0; // a finished run starts again
		playing = finished || !playing;
	}

	// The simulations so far: their mean, and two standard errors around it.
	const runs = $derived(data ? data.simulated.slice(0, shown).map((v, i) => ({ i: i + 1, v })) : []);
	const stats = $derived.by(() => {
		const out: { i: number; mean: number; lo: number; hi: number }[] = [];
		let sum = 0;
		let sq = 0;
		runs.forEach(({ i, v }) => {
			sum += v;
			sq += v * v;
			const mean = sum / i;
			const sd = i > 1 ? Math.sqrt(Math.max(0, (sq - i * mean * mean) / (i - 1))) : 0;
			const se = sd / Math.sqrt(i);
			out.push({ i, mean, lo: mean - 2 * se, hi: mean + 2 * se });
		});
		return out;
	});
	const last = $derived(stats.at(-1));

	$effect(() => {
		themeTick;
		if (!data || !chart) return;
		const c = themeColors();
		const all = data.simulated;
		const lo = Math.min(...all, data.computed - 2.2 * data.sd);
		const hi = Math.max(...all, data.computed + 2.2 * data.sd);
		const pad = (hi - lo) * 0.08;
		const band = (k: number) => [
			{ i: 0, lo: data!.computed - k * data!.sd, hi: data!.computed + k * data!.sd },
			{ i: all.length + 1, lo: data!.computed - k * data!.sd, hi: data!.computed + k * data!.sd },
		];
		const plot = Plot.plot({
			width: chart.clientWidth,
			height: 250,
			marginLeft: 64,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { label: 'Simulation →', domain: [0, all.length + 1] },
			y: {
				label: '↑ Distinct sequences in 100,000 cells',
				domain: [lo - pad, hi + pad],
				grid: true,
				tickFormat: (v: number) => fmt(v),
			},
			marks: [
				// Where BEAM says one simulation should land: within 1 and 2 standard deviations.
				Plot.areaY(band(2), { x: 'i', y1: 'lo', y2: 'hi', fill: c.accent, fillOpacity: 0.1 }),
				Plot.areaY(band(1), { x: 'i', y1: 'lo', y2: 'hi', fill: c.accent, fillOpacity: 0.14 }),
				Plot.areaY(stats.filter((d) => d.i > 1), { x: 'i', y1: 'lo', y2: 'hi', fill: c.second, fillOpacity: 0.15 }),
				Plot.line(stats, { x: 'i', y: 'mean', stroke: c.second, strokeWidth: 1.5 }),
				Plot.dot(runs, { x: 'i', y: 'v', fill: c.muted, r: 2.5 }),
				Plot.ruleY([data.computed], { stroke: c.accent, strokeWidth: 2.5 }),
			],
		});
		chart.replaceChildren(plot);
	});
</script>

<div class="simulated">
	{#if data}
		<div class="controls">
			<button onclick={toggle}>
				{playing ? 'Pause' : finished ? 'Run again' : shown ? 'Resume' : 'Run simulations'}
			</button>
			<button
				onclick={() => {
					playing = false;
					shown = 0;
				}}>Reset</button
			>
			<span class="count" aria-live="polite"><strong>{shown}</strong> of {data.simulated.length} simulations</span>
		</div>
		<div
			bind:this={chart}
			role="img"
			aria-label="Simulated counts of distinct sequences in 100,000 cells, their running mean, and BEAM's computed value as a fixed line"
		></div>
		<div class="legend">
			<span><span class="swatch a"></span>computed by BEAM, once</span>
			<span><span class="swatch band"></span>computed range of one simulation: ± 1 and ± 2 standard deviations</span>
			<span><span class="swatch dot"></span>one simulation of 100,000 cells</span>
			<span><span class="swatch b"></span>mean of the simulations so far, ± 2 standard errors</span>
		</div>
		<p class="readout" aria-live="polite">
			BEAM: <strong>{fmt(data.computed, 1)} ± {fmt(data.sd, 1)}</strong> distinct sequences (one standard
			deviation of a single sample), both computed, with guaranteed relative errors below {sci(data.bound)}
			and {sci(data.sd_bound)}.
			{#if last && last.i > 1}
				After {last.i} simulations: {fmt(last.mean, 1)} ± {fmt((last.hi - last.lo) / 2, 1)}; single
				simulations ranged from {fmt(Math.min(...data.simulated.slice(0, shown)))} to {fmt(
					Math.max(...data.simulated.slice(0, shown)),
				)}.
			{/if}
		</p>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.simulated {
		display: grid;
		gap: 0.5rem;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		align-items: center;
	}
	button {
		font: inherit;
		font-size: var(--sl-text-sm);
		padding: 0.25rem 0.8rem;
		border-radius: 999px;
		border: 1px solid var(--sl-color-gray-5);
		background: transparent;
		color: var(--sl-color-white);
		cursor: pointer;
	}
	button:first-child {
		background: var(--sl-color-accent);
		border-color: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	.count {
		font-size: var(--sl-text-sm);
		color: var(--sl-color-gray-2);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem 1.1rem;
		font-size: var(--sl-text-xs);
		color: var(--sl-color-gray-2);
	}
	.swatch {
		display: inline-block;
		width: 0.9rem;
		height: 0.2rem;
		margin-inline-end: 0.35rem;
		vertical-align: middle;
		border-radius: 2px;
	}
	.swatch.a {
		background: var(--sl-color-accent);
		height: 0.25rem;
	}
	.swatch.band {
		height: 0.7rem;
		background: color-mix(in srgb, var(--sl-color-accent) 30%, transparent);
	}
	.swatch.b {
		background: #ef785f;
	}
	.swatch.dot {
		width: 0.45rem;
		height: 0.45rem;
		border-radius: 50%;
		background: var(--sl-color-gray-3);
	}
	.readout {
		font-size: var(--sl-text-sm);
		margin: 0;
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
