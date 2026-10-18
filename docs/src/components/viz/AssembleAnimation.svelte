<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, onThemeChange, sci, themeColors } from './theme';

	interface Assemble {
		sequence: string;
		forward: number[];
		backward: number[];
		combined: number[];
		log_length_prior: number;
		log_partition: number;
		log_pseq: number;
	}

	let a = $state<Assemble | null>(null);
	let shown = $state(0); // residues assembled so far; L + 1 adds the length terms
	let playing = $state(false);
	let themeTick = $state(0);
	let failed = $state(false);
	let chart: HTMLDivElement | undefined = $state();

	$effect(() => {
		fetch(dataUrl('viz/model_parts/model_parts.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d) => (a = d.assemble))
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	const L = $derived(a ? a.sequence.length : 0);
	const done = $derived(a !== null && shown > L);
	const partial = $derived(a ? a.combined.slice(0, Math.min(shown, L)).reduce((s, v) => s + v, 0) : 0);

	$effect(() => {
		if (!playing || !a) return;
		const timer = setInterval(() => {
			if (shown > L) playing = false;
			else shown += 1;
		}, 650);
		return () => clearInterval(timer);
	});

	function toggle() {
		if (done) shown = 0;
		playing = done || !playing;
	}

	const LN2 = Math.log(2);
	$effect(() => {
		themeTick;
		if (!a || !chart) return;
		const c = themeColors();
		const n = Math.min(shown, L);
		const rows = Array.from({ length: n }, (_, i) => ({
			i: i + 1,
			f: -a!.forward[i] / LN2,
			b: -a!.backward[i] / LN2,
			psi: -a!.combined[i] / LN2,
		}));
		const plot = Plot.plot({
			width: chart.clientWidth,
			height: 220,
			marginLeft: 48,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: {
				domain: [0.4, L + 0.6],
				label: null,
				ticks: Array.from({ length: L }, (_, i) => i + 1),
				tickFormat: (v: number) => a!.sequence[v - 1] ?? '',
			},
			y: {
				domain: [0, 1.1 * Math.max(...[...a.forward, ...a.backward].map((v) => -v / LN2))],
				label: '↑ surprisal (bits)',
				grid: true,
			},
			marks: [
				Plot.rectY(rows, { x1: (d) => d.i - 0.38, x2: (d) => d.i, y: 'f', fill: c.accent, fillOpacity: 0.85 }),
				Plot.rectY(rows, { x1: (d) => d.i, x2: (d) => d.i + 0.38, y: 'b', fill: c.second, fillOpacity: 0.85 }),
				Plot.line(rows, { x: 'i', y: 'psi', stroke: '#6c7bd9', strokeWidth: 2 }),
				Plot.dot(rows, { x: 'i', y: 'psi', fill: '#6c7bd9', r: 3.5 }),
				Plot.ruleY([0], { stroke: c.muted }),
			],
		});
		chart.replaceChildren(plot);
	});
</script>

<div class="assemble">
	{#if a}
		<div class="controls">
			<button class="primary" onclick={toggle}>{playing ? 'Pause' : done ? 'Assemble again' : shown ? 'Resume' : 'Assemble'}</button>
			<button onclick={() => ((playing = false), (shown = Math.min(L + 1, shown + 1)))}>Step ▶</button>
			<button onclick={() => ((playing = false), (shown = 0))}>Reset</button>
		</div>
		<div class="seq mono" aria-label="Sequence being scored">
			{#each [...a.sequence] as ch, i}
				<span class="res" class:done={i < shown} class:here={i === shown - 1 && shown <= L}>{ch}</span>
			{/each}
		</div>
		<div bind:this={chart} role="img" aria-label="Surprisal of each residue under the forward and backward experts, and the combined factor"></div>
		<div class="legend">
			<span><span class="swatch f"></span>forward expert, −log₂ q<sub>F</sub></span>
			<span><span class="swatch b"></span>backward expert, −log₂ q<sub>B</sub></span>
			<span><span class="swatch psi"></span>combined, −log₂ ψ<sub>i</sub></span>
		</div>
		<div class="sum mono" aria-live="polite">
			{#if shown === 0}
				<span class="muted">Press Assemble to multiply the residues' factors one by one.</span>
			{:else if shown <= L}
				<span>residue {shown} ({a.sequence[shown - 1]}): ln ψ = {a.combined[shown - 1].toFixed(4)}</span>
				<span>Σ ln ψ so far = <strong>{partial.toFixed(4)}</strong></span>
			{:else}
				<span>Σ ln ψ = {partial.toFixed(4)}</span>
				<span>+ ln π(L) = {a.log_length_prior.toFixed(4)}  − ln Z(L) = {(-a.log_partition).toFixed(4)}</span>
				<span>ln P(s) = <strong>{(partial + a.log_length_prior - a.log_partition).toFixed(4)}</strong>, as log_pseq returns
					({a.log_pseq.toFixed(4)}); P(s) = {sci(Math.exp(a.log_pseq), 3)}</span>
			{/if}
		</div>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.assemble {
		display: grid;
		gap: 0.55rem;
		font-size: var(--sl-text-sm);
	}
	.controls {
		display: flex;
		gap: 0.5rem;
	}
	button {
		font: inherit;
		font-size: var(--sl-text-sm);
		padding: 0.2rem 0.75rem;
		border-radius: 999px;
		border: 1px solid var(--sl-color-gray-5);
		background: transparent;
		color: var(--sl-color-white);
		cursor: pointer;
	}
	button.primary {
		background: var(--sl-color-accent);
		border-color: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	.seq {
		display: flex;
		gap: 3px;
	}
	.res {
		display: inline-grid;
		place-items: center;
		width: 1.6rem;
		height: 1.6rem;
		border-radius: 0.3rem;
		background: var(--sl-color-gray-6);
		opacity: 0.5;
		transition: opacity 0.2s ease;
	}
	.res.done {
		opacity: 1;
	}
	.res.here {
		background: var(--sl-color-white);
		color: var(--sl-color-black);
		font-weight: 700;
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
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 2px;
		margin-inline-end: 0.3rem;
		vertical-align: -0.1rem;
	}
	.swatch.f {
		background: var(--sl-color-accent);
	}
	.swatch.b {
		background: #ef785f;
	}
	.swatch.psi {
		background: #6c7bd9;
	}
	.sum {
		display: grid;
		gap: 0.2rem;
		font-size: var(--sl-text-xs);
		padding: 0.5rem 0.7rem;
		border-left: 3px solid var(--sl-color-accent);
		background: var(--sl-color-gray-7, var(--sl-color-black));
		min-height: 3.6rem;
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
