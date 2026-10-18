<script lang="ts">
	import { dataUrl, sci } from './theme';

	interface Toy {
		train: string[];
		probes: string[];
		k: number;
		alpha: number[];
		explained: Record<string, { forward: number[]; backward: number[] }>;
	}

	let toy = $state<Toy | null>(null);
	let seq = $state('');
	let pos = $state(12);
	let backward = $state(false);
	let logMult = $state(0); // prior strength multiplier, as log10
	let failed = $state(false);

	$effect(() => {
		fetch(dataUrl('viz/model_parts/model_parts.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d) => {
				toy = d.toy;
				seq = d.toy.probes[0];
				pos = d.toy.probes[0].length - 1;
			})
			.catch(() => (failed = true));
	});

	const mult = $derived(10 ** logMult);
	const choices = $derived(toy ? [...toy.probes, ...toy.train.slice(0, 3)] : []);

	// The chain for one residue, from the uniform 1/20 up to the full context of k residues:
	// P_j(x) = (n_j(x) + alpha_j * P_{j-1}(x)) / (n_j + alpha_j).
	const chain = $derived.by(() => {
		if (!toy || !seq) return null;
		const k = toy.k;
		const rev = (s: string) => [...s].reverse().join('');
		const r = backward ? rev(seq) : seq;
		const i = backward ? seq.length - 1 - pos : pos;
		const x = r[i];
		const reads = toy.train.map((s) => (backward ? rev(s) : s));
		const levels: {
			j: number;
			ctx: string;
			n: number;
			nx: number;
			alpha: number;
			fromCounts: number;
			fromShorter: number;
			p: number;
		}[] = [];
		let p = 1 / 20;
		for (let j = 0; j <= k; j++) {
			const ctx = r.slice(Math.max(0, i - j), i).padStart(j, '·');
			let n = 0;
			let nx = 0;
			for (const t of reads) {
				if (i < t.length && t.slice(Math.max(0, i - j), i).padStart(j, '·') === ctx) {
					n += 1;
					if (t[i] === x) nx += 1;
				}
			}
			const a = toy.alpha[j] * mult;
			const fromCounts = nx / (n + a);
			const fromShorter = (a * p) / (n + a);
			p = fromCounts + fromShorter;
			levels.push({ j, ctx, n, nx, alpha: a, fromCounts, fromShorter, p });
		}
		const ex = toy.explained[seq];
		const beam = ex ? Math.exp((backward ? ex.backward : ex.forward)[pos]) : null;
		return { x, position: i + 1, levels, beam };
	});
</script>

<div class="prior">
	{#if toy && chain}
		<div class="row" role="group" aria-label="Sequence">
			<span class="muted">Sequence</span>
			{#each choices as s}
				<button class="mono" class:active={seq === s} aria-pressed={seq === s} onclick={() => ((seq = s), (pos = Math.min(pos, s.length - 1)))}>{s}</button>
			{/each}
		</div>
		<div class="row" role="group" aria-label="Residue">
			<span class="muted">Residue</span>
			{#each [...seq] as ch, i}
				<button class="res mono" class:active={pos === i} aria-pressed={pos === i} onclick={() => (pos = i)}>{ch}</button>
			{/each}
		</div>
		<div class="row" role="group" aria-label="Expert">
			<span class="muted">Expert</span>
			<button class:active={!backward} aria-pressed={!backward} onclick={() => (backward = false)}>forward</button>
			<button class:active={backward} aria-pressed={backward} onclick={() => (backward = true)}>backward</button>
		</div>
		<label class="slider">
			<span class="muted">Prior strength</span>
			<input type="range" min="-1.5" max="1.5" step="0.05" bind:value={logMult} aria-label="Multiply every prior strength alpha" />
			<span>× {mult < 1 ? mult.toFixed(2) : mult.toFixed(1)}{Math.abs(logMult) < 0.03 ? ' (fitted by BEAM)' : ''}</span>
		</label>

		<div class="chain" aria-live="polite">
			<div class="level base">
				<span class="lab">below every level</span>
				<span>every residue equally likely: 1/20 = 0.05</span>
			</div>
			{#each chain.levels as l}
				<div class="level">
					<span class="lab">
						{l.j === 0 ? 'position only' : `${l.j} residue${l.j > 1 ? 's' : ''} of context`}
						<span class="mono ctx">{l.j ? l.ctx : '—'}</span>
					</span>
					<span class="counts">
						seen {l.n}×, followed by <span class="mono">{chain.x}</span> {l.nx}×; α = {l.alpha.toFixed(2)}
					</span>
					<span class="bar" title={`${((100 * l.fromCounts) / l.p).toFixed(0)}% from this level's counts, ${((100 * l.fromShorter) / l.p).toFixed(0)}% inherited`}>
						<span class="seg own" style:width={`${(100 * l.fromCounts) / l.p}%`}></span>
						<span class="seg inh" style:width={`${(100 * l.fromShorter) / l.p}%`}></span>
					</span>
					<span class="p">P = {l.p < 1e-3 ? sci(l.p) : l.p.toFixed(4)}</span>
				</div>
			{/each}
		</div>
		<p class="readout">
			<span class="mono">{chain.x}</span> at position {chain.position} from the {backward ? 'end' : 'start'}:
			q = <strong>{chain.levels[chain.levels.length - 1].p.toFixed(6)}</strong>
			{#if chain.beam !== null && Math.abs(logMult) < 0.03}
				— BEAM's own value, from <span class="mono">explain</span>: {chain.beam.toFixed(6)}.
			{:else}
				(at the fitted strength, BEAM's own value is {chain.beam?.toFixed(6)}).
			{/if}
		</p>
		<div class="legend">
			<span>each bar is one level's P, split into:</span>
			<span><span class="swatch own"></span>the share from its own counts</span>
			<span><span class="swatch inh"></span>the share inherited from the shorter context</span>
		</div>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.prior {
		display: grid;
		gap: 0.55rem;
		font-size: var(--sl-text-sm);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		align-items: center;
	}
	.row > .muted,
	.slider > .muted {
		min-width: 6.5rem;
	}
	button {
		font: inherit;
		font-size: var(--sl-text-xs);
		padding: 0.15rem 0.55rem;
		border-radius: 999px;
		border: 1px solid var(--sl-color-gray-5);
		background: transparent;
		color: var(--sl-color-white);
		cursor: pointer;
	}
	button.res {
		border-radius: 0.25rem;
		padding: 0.1rem 0.35rem;
	}
	button.active {
		background: var(--sl-color-accent);
		border-color: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	.slider {
		display: flex;
		align-items: center;
		gap: 0.6rem;
	}
	.slider input {
		flex: 1;
		accent-color: var(--sl-color-accent);
	}
	.chain {
		display: flex;
		flex-direction: column-reverse;
		gap: 0.3rem;
	}
	.level {
		display: grid;
		grid-template-columns: minmax(9rem, 1.2fr) minmax(10rem, 1.4fr) 1fr 6rem;
		gap: 0.6rem;
		align-items: center;
		padding: 0.3rem 0.5rem;
		border-radius: 0.35rem;
		background: var(--sl-color-gray-7, var(--sl-color-black));
		font-size: var(--sl-text-xs);
	}
	.level.base {
		grid-template-columns: minmax(9rem, 1.2fr) 1fr;
		color: var(--sl-color-gray-3);
	}
	.bar {
		display: flex;
		height: 0.7rem;
		border-radius: 0.2rem;
		overflow: hidden;
		background: var(--sl-color-gray-6);
	}
	.seg {
		height: 100%;
		transition: width 0.25s ease;
	}
	.own,
	.swatch.own {
		background: var(--sl-color-accent);
	}
	.inh,
	.swatch.inh {
		background: #6c7bd9;
	}
	.p {
		font-family: var(--sl-font-mono);
		text-align: right;
	}
	.ctx {
		margin-inline-start: 0.3rem;
		padding: 0 0.3rem;
		border-radius: 0.2rem;
		background: var(--sl-color-gray-6);
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.readout {
		margin: 0;
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
	.muted {
		color: var(--sl-color-gray-3);
	}
	@media (max-width: 40rem) {
		.level {
			grid-template-columns: 1fr 1fr;
		}
	}
</style>
