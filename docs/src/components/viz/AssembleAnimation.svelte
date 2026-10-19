<script lang="ts">
	import { dataUrl, sci } from './theme';
	import SequenceLens from './SequenceLens.svelte';

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
	let shown = $state(0); // rows of the product revealed: the length, each residue, then the normaliser
	let playing = $state(false);
	let failed = $state(false);
	let width = $state(640);

	$effect(() => {
		fetch(dataUrl('viz/model_parts/model_parts.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d) => (a = d.assemble))
			.catch(() => (failed = true));
	});

	const L = $derived(a ? a.sequence.length : 0);
	const total = $derived(L + 2);
	const done = $derived(a !== null && shown >= total);

	$effect(() => {
		if (!playing || !a) return;
		const timer = setInterval(() => {
			if (shown >= total) playing = false;
			else shown += 1;
		}, 700);
		return () => clearInterval(timer);
	});

	function toggle() {
		if (done) shown = 0;
		playing = done || !playing;
	}

	// One row per factor of P(s), each a step on a log scale from the running value before it to
	// the one after: the length's probability, every residue's factor, and division by Z(L).
	const LN10 = Math.log(10);
	const rows = $derived.by(() => {
		if (!a) return [];
		const out: { label: string; factor: string; from: number; to: number; kind: string; residue: number | null }[] = [];
		let v = 0;
		out.push({ label: `length ${L}`, factor: `× π(${L}) = ${Math.exp(a.log_length_prior).toFixed(3)}`, from: 0, to: a.log_length_prior, kind: 'len', residue: null });
		v = a.log_length_prior;
		a.combined.forEach((c, i) => {
			out.push({ label: `${i + 1}  ${a!.sequence[i]}`, factor: `× ψ = ${Math.exp(c).toFixed(3)}`, from: v, to: v + c, kind: 'psi', residue: i });
			v += c;
		});
		out.push({ label: 'normaliser', factor: `÷ Z(${L}) = ${Math.exp(a.log_partition).toFixed(3)}`, from: v, to: v - a.log_partition, kind: 'z', residue: null });
		return out;
	});
	const current = $derived(shown > 0 ? rows[shown - 1] : null);
	const lowest = $derived(Math.min(...rows.map((r) => r.to)) / LN10);
	const decades = $derived(Math.ceil(-lowest));

	const labelW = 76;
	const factorW = 104;
	const rowH = 17;
	const top = 36;
	const plotW = $derived(Math.max(160, width - labelW - factorW - 8));
	const xOf = (lnv: number) => labelW + (plotW * (-lnv / LN10)) / decades;
</script>

<div class="assemble" bind:clientWidth={width}>
	{#if a}
		<div class="controls">
			<button class="primary" onclick={toggle}>{playing ? 'Pause' : done ? 'Assemble again' : shown ? 'Resume' : 'Assemble'}</button>
			<button onclick={() => ((playing = false), (shown = Math.min(total, shown + 1)))}>Step ▶︎</button>
			<button onclick={() => ((playing = false), (shown = 0))}>Reset</button>
		</div>

		<SequenceLens
			sequence={a.sequence}
			k={4}
			windows={current?.residue != null ? [{ target: current.residue, dir: 'f' }, { target: current.residue, dir: 'b' }] : []}
		/>

		<svg width={labelW + plotW + factorW} height={top + rows.length * rowH + 6} role="img"
			aria-label="The probability of CASSLGQGYEQYF built up factor by factor on a log scale: the length's probability, each residue's factor, then division by the normaliser">
			<text x={labelW} y={10} class="axis">running value, on a log scale →</text>
			{#each Array(decades + 1) as _, d}
				<line x1={labelW + (plotW * d) / decades} x2={labelW + (plotW * d) / decades} y1={top - 4} y2={top + rows.length * rowH} class="grid" />
				<text x={labelW + (plotW * d) / decades} y={top - 8} text-anchor="middle" class="tick">{d === 0 ? '1' : `10⁻${'⁰¹²³⁴⁵⁶⁷⁸⁹'[d]}`}</text>
			{/each}
			{#each rows as r, j}
				{@const y = top + j * rowH}
				<g class="row {r.kind}" class:hidden={j >= shown} class:now={j === shown - 1}>
					<text x={labelW - 8} y={y + rowH - 5} text-anchor="end" class="lab">{r.label}</text>
					<rect x={Math.min(xOf(r.from), xOf(r.to))} y={y + 3} width={Math.max(1.5, Math.abs(xOf(r.to) - xOf(r.from)))} height={rowH - 6} rx="2" class="seg" />
					{#if j > 0}
						<line x1={xOf(r.from)} x2={xOf(r.from)} y1={y - 3} y2={y + 3} class="link" />
					{/if}
					<text x={labelW + plotW + 6} y={y + rowH - 5} class="fac">{r.factor}</text>
				</g>
			{/each}
			{#if done}
				<circle cx={xOf(rows[rows.length - 1].to)} cy={top + (rows.length - 0.5) * rowH} r="4" class="end" />
			{/if}
		</svg>

		<div class="sum mono" aria-live="polite">
			{#if !current}
				<span class="muted">Press Assemble to build P(s) one factor at a time, on a log scale: each factor below one moves it to the right.</span>
			{:else if current.kind === 'len'}
				<span>start from the probability of the length: π({L}) = {Math.exp(a.log_length_prior).toFixed(4)}</span>
			{:else if current.kind === 'psi'}
				<span>residue {shown - 1} ({a.sequence[shown - 2]}): × ψ = {Math.exp(current.to - current.from).toFixed(4)}, the two experts' weighted geometric mean</span>
				<span>running value π({L}) × ψ₁ ⋯ ψ<sub>{shown - 1}</sub> = <strong>{sci(Math.exp(current.to), 3)}</strong></span>
			{:else}
				<span>divide by the normaliser Z({L}) = {Math.exp(a.log_partition).toFixed(4)}, so that all length-{L} sequences together get π({L})</span>
				<span>P(s) = <strong>{sci(Math.exp(current.to), 3)}</strong>; ln P(s) = {current.to.toFixed(4)}, as log_pseq returns ({a.log_pseq.toFixed(4)})</span>
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
		background: var(--sl-color-white);
		border-color: var(--sl-color-white);
		color: var(--sl-color-black);
	}
	svg {
		max-width: 100%;
		overflow: visible;
	}
	.grid {
		stroke: var(--sl-color-gray-5);
		stroke-dasharray: 2 3;
	}
	.axis {
		font-size: 11px;
		font-style: italic;
		fill: var(--sl-color-gray-3);
	}
	.tick {
		font-size: 11px;
		fill: var(--sl-color-gray-3);
	}
	.row {
		transition: opacity 0.3s ease;
	}
	.row.hidden {
		opacity: 0;
	}
	.lab,
	.fac {
		font-size: 11px;
		font-family: var(--sl-font-mono);
		fill: var(--sl-color-gray-2);
	}
	.row.now .lab,
	.row.now .fac {
		fill: var(--sl-color-white);
		font-weight: 700;
	}
	.seg {
		fill: #6c7bd9;
	}
	.row.len .seg {
		fill: var(--sl-color-gray-3);
	}
	.row.z .seg {
		fill: var(--sl-color-accent);
	}
	.link {
		stroke: var(--sl-color-gray-3);
	}
	.end {
		fill: var(--sl-color-white);
	}
	.sum {
		display: grid;
		gap: 0.2rem;
		font-size: var(--sl-text-xs);
		padding: 0.5rem 0.7rem;
		border-left: 3px solid #6c7bd9;
		background: var(--sl-color-gray-7, var(--sl-color-black));
		min-height: 3.2rem;
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
