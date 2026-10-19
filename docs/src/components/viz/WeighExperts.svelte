<script lang="ts">
	import { dataUrl, sci } from './theme';
	import SequenceLens from './SequenceLens.svelte';

	interface Walk {
		sequence: string;
		alphabet: string;
		k: number;
		weights: [number, number];
		forward: number[][];
		backward: number[][];
	}

	let w = $state<Walk | null>(null);
	let i = $state(12);
	let cand = $state('F'); // the residue whose factor is read out
	let wf = $state(0.5);
	let failed = $state(false);

	$effect(() => {
		fetch(dataUrl('viz/expert_walk/expert_walk.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: Walk) => {
				w = d;
				wf = d.weights[0];
				// Start on the residue where the two experts disagree most.
				const gap = [...d.sequence].map((x, j) => {
					const a = d.alphabet.indexOf(x);
					return Math.abs(Math.log(d.forward[j][a] / d.backward[j][a]));
				});
				i = gap.indexOf(Math.max(...gap));
				cand = d.sequence[i];
			})
			.catch(() => (failed = true));
	});

	function pick(j: number) {
		i = j;
		cand = w!.sequence[j];
	}

	const wb = $derived(1 - wf);
	const L = $derived(w ? w.sequence.length : 0);
	const qf = $derived(w ? w.forward[i] : []);
	const qb = $derived(w ? w.backward[i] : []);
	const psi = $derived(qf.map((f, j) => f ** wf * qb[j] ** wb));
	const c = $derived(w ? w.alphabet.indexOf(cand) : 0);
	const rows = $derived([
		{ key: 'f', name: 'Forward expert', sym: 'q', sub: 'F', values: qf },
		{ key: 'b', name: 'Backward expert', sym: 'q', sub: 'B', values: qb },
		{ key: 'psi', name: 'Combined factor', sym: 'ψ', sub: '', values: psi },
	]);
	const num = (p: number) => (p >= 0.001 ? p.toFixed(3) : sci(p, 2));
	const base = (p: number) => (p >= 0.001 ? num(p) : `(${num(p)})`); // bracketed before an exponent
	// Which expert finds the residue far likelier than the other, if either does.
	const keener = $derived(qf[c] / qb[c] > 10 ? 'forward' : qb[c] / qf[c] > 10 ? 'backward' : null);
	const ratio = $derived(Math.max(qf[c] / qb[c], qb[c] / qf[c]));
</script>

<div class="weigh">
	{#if w}
		<p class="hint muted">Choose a residue of the sequence, then a column to ask about any amino acid in its place.</p>
		<SequenceLens sequence={w.sequence} k={w.k} windows={[{ target: i, dir: 'f' }, { target: i, dir: 'b' }]} onpick={pick} />

		<div class="rows">
			{#each rows as r}
				<div class="row {r.key}">
					<span class="name">{r.name}<br /><span class="sym">{r.sym}<sub>{r.sub}</sub></span></span>
					<div class="bars">
						{#each [...w.alphabet] as a, j}
							<button class="col" class:hit={j === c} onclick={() => (cand = a)} aria-label={`${a}: ${r.sym}${r.sub} = ${num(r.values[j])}`}>
								<span class="bar" style:height={`${Math.max(1, 100 * r.values[j])}%`}></span>
							</button>
						{/each}
					</div>
				</div>
			{/each}
			<div class="row letters" aria-hidden="true">
				<span></span>
				<div class="bars">
					{#each [...w.alphabet] as a, j}<span class="aa" class:hit={j === c}>{a}</span>{/each}
				</div>
			</div>
		</div>

		<label class="slider">
			<span>weight of the forward expert, w<sub>F</sub> = <strong>{wf.toFixed(2)}</strong>; of the backward, w<sub>B</sub> = <strong>{wb.toFixed(2)}</strong></span>
			<input type="range" min="0" max="1" step="0.05" bind:value={wf} aria-label="Weight of the forward expert" />
			<span class="ticks muted"><span>backward alone</span><span>BEAM's default, ½ each</span><span>forward alone</span></span>
		</label>

		<div class="eq mono" aria-live="polite">
			<span>
				<span class="mono">{cand}</span> as residue {i + 1} of {L}: ψ = q<sub>F</sub><sup>{wf.toFixed(2)}</sup> × q<sub>B</sub><sup>{wb.toFixed(2)}</sup>
				= {base(qf[c])}<sup>{wf.toFixed(2)}</sup> × {base(qb[c])}<sup>{wb.toFixed(2)}</sup> = <strong>{num(psi[c])}</strong>
			</span>
			<span class="plain">
				{#if Math.max(qf[c], qb[c]) < 0.01}
					Both experts find {cand} unlikely here, so its factor is small too.
				{:else if keener}
					The {keener} expert finds {cand} {ratio < 100 ? ratio.toFixed(0) : Math.round(ratio).toLocaleString('en-US')} times
					likelier than the {keener === 'forward' ? 'backward' : 'forward'} expert does: the factor lies between the two, pulled
					down by the expert that objects.
				{:else}
					The two experts roughly agree on {cand} here, so its factor is close to both.
				{/if}
			</span>
		</div>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.weigh {
		--f: var(--sl-color-accent);
		--b: #ef785f;
		--psi: #6c7bd9;
		display: grid;
		gap: 0.6rem;
		font-size: var(--sl-text-sm);
	}
	.hint {
		margin: 0;
		font-size: var(--sl-text-xs);
	}
	.rows {
		display: grid;
		gap: 0.35rem;
	}
	.row {
		display: grid;
		grid-template-columns: 6.5rem 1fr;
		gap: 0.5rem;
		align-items: end;
	}
	.name {
		font-size: var(--sl-text-xs);
		line-height: 1.2;
		color: var(--sl-color-gray-2);
		padding-bottom: 0.2rem;
	}
	.sym {
		font-family: var(--sl-font-mono);
		font-weight: 700;
		color: var(--c);
	}
	.row.f {
		--c: var(--f);
	}
	.row.b {
		--c: var(--b);
	}
	.row.psi {
		--c: var(--psi);
	}
	.bars {
		display: grid;
		grid-template-columns: repeat(20, 1fr);
		gap: 3px;
		height: 3.6rem;
		align-items: end;
		border-bottom: 1px solid var(--sl-color-gray-5);
	}
	.row.letters .bars {
		height: auto;
		border: 0;
	}
	.col {
		height: 100%;
		padding: 0;
		border: 0;
		background: transparent;
		display: flex;
		align-items: flex-end;
		justify-content: center;
		cursor: pointer;
		border-radius: 3px 3px 0 0;
	}
	.col:hover {
		background: var(--sl-color-gray-6);
	}
	.col.hit {
		background: color-mix(in srgb, var(--sl-color-white) 12%, transparent);
	}
	.bar {
		width: 100%;
		max-width: 1.6rem;
		background: var(--c);
		opacity: 0.7;
		border-radius: 3px 3px 0 0;
		transition: height 0.3s ease;
	}
	.col.hit .bar {
		opacity: 1;
	}
	.aa {
		text-align: center;
		font-family: var(--sl-font-mono);
		font-size: var(--sl-text-xs);
		color: var(--sl-color-gray-3);
	}
	.aa.hit {
		color: var(--sl-color-white);
		font-weight: 700;
	}
	.slider {
		display: grid;
		gap: 0.2rem;
	}
	.slider input {
		width: 100%;
		accent-color: var(--psi);
	}
	.ticks {
		display: flex;
		justify-content: space-between;
		font-size: var(--sl-text-xs);
	}
	.eq {
		display: grid;
		gap: 0.3rem;
		font-size: var(--sl-text-xs);
		padding: 0.5rem 0.7rem;
		border-left: 3px solid var(--psi);
		background: var(--sl-color-gray-7, var(--sl-color-black));
	}
	.plain {
		font-family: var(--sl-font-system);
		color: var(--sl-color-gray-2);
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
	@media (max-width: 40rem) {
		.row {
			grid-template-columns: 4.5rem 1fr;
		}
	}
</style>
