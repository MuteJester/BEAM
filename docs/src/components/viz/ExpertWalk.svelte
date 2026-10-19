<script lang="ts">
	import { dataUrl } from './theme';
	import SequenceLens from './SequenceLens.svelte';

	interface Walk {
		sequence: string;
		alphabet: string;
		k: number;
		forward: number[][];
		backward: number[][];
	}

	let w = $state<Walk | null>(null);
	let n = $state(0); // two beats per step: the experts predict, then the residues are revealed
	let playing = $state(false);
	let failed = $state(false);

	$effect(() => {
		fetch(dataUrl('viz/expert_walk/expert_walk.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: Walk) => (w = d))
			.catch(() => (failed = true));
	});

	const L = $derived(w ? w.sequence.length : 0);
	const last = $derived(2 * L - 1);
	const step = $derived(Math.floor(n / 2));
	const revealed = $derived(n % 2 === 1);

	$effect(() => {
		if (!playing || !w) return;
		const timer = setInterval(() => {
			if (n >= last) playing = false;
			else n += 1;
		}, 900);
		return () => clearInterval(timer);
	});

	function toggle() {
		if (n >= last) n = 0;
		playing = n >= last || !playing;
	}

	const ordinal = (m: number) =>
		`${m}${m % 10 === 1 && m !== 11 ? 'st' : m % 10 === 2 && m !== 12 ? 'nd' : m % 10 === 3 && m !== 13 ? 'rd' : 'th'}`;

	// Both experts take a step at once: the forward expert from the first residue, the backward
	// expert from the last, so they cross in the middle.
	function view(dir: 'f' | 'b') {
		const d = w!;
		const i = dir === 'f' ? step : L - 1 - step;
		const row = (dir === 'f' ? d.forward : d.backward)[i];
		const x = d.sequence[i];
		const q = row[d.alphabet.indexOf(x)];
		const ranked = [...d.alphabet].map((a, j) => ({ a, p: row[j] })).sort((u, v) => v.p - u.p);
		// What it has just read: the k residues before (forward) or after (backward), in sequence
		// order, with '·' past the end of the sequence.
		const context: string[] = [];
		for (let j = 1; j <= d.k; j++) {
			const p = dir === 'f' ? i - d.k - 1 + j : i + j;
			context.push(p >= 0 && p < L ? d.sequence[p] : '·');
		}
		return { dir, i, row, x, q, ranked, rank: ranked.findIndex((r) => r.a === x) + 1, context, empty: context.every((c) => c === '·') };
	}
	const experts = $derived(w ? [view('f'), view('b')] : []);

	// The probability each expert gave every residue it has already passed.
	function passed(dir: 'f' | 'b') {
		const d = w!;
		return [...d.sequence].map((ch, p) => {
			const s = dir === 'f' ? p : L - 1 - p;
			const done = s < step || (s === step && revealed);
			return done ? (dir === 'f' ? d.forward : d.backward)[p][d.alphabet.indexOf(ch)] : null;
		});
	}
	const strips = $derived(
		w
			? [
					{ dir: 'f' as const, values: passed('f'), label: 'forward gave' },
					{ dir: 'b' as const, values: passed('b'), label: 'backward gave' },
				]
			: [],
	);
</script>

<div class="walk">
	{#if w}
		<div class="controls">
			<button class="primary" onclick={toggle}>{playing ? 'Pause' : n >= last ? 'Walk again' : n ? 'Resume' : 'Walk'}</button>
			<button onclick={() => ((playing = false), (n = Math.min(last, n + 1)))}>Step ▶︎</button>
			<button onclick={() => ((playing = false), (n = 0))}>Reset</button>
			<span class="muted">step {step + 1} of {L}</span>
		</div>

		<SequenceLens
			sequence={w.sequence}
			k={w.k}
			windows={[{ target: experts[0].i, dir: 'f' }, { target: experts[1].i, dir: 'b' }]}
			hidden={!revealed}
			{strips}
		/>

		<div class="cards">
			{#each experts as e (e.dir)}
				<div class="card {e.dir}">
					<div class="ask" aria-live="polite">
						<span class="who">{e.dir === 'f' ? 'Forward expert →' : '← Backward expert'}</span>
						<span>
							I am at the <strong>{ordinal(e.dir === 'f' ? e.i + 1 : L - e.i)}</strong> residue from the {e.dir === 'f' ? 'start' : 'end'}{#if e.empty},
								with nothing read yet.
							{:else}, and I have just read
								<span class="chips">{#each e.context as c}<span class="chip" class:pad={c === '·'}>{c}</span>{/each}</span>.
							{/if}
							Which residue comes {e.dir === 'f' ? 'next' : 'before'}?
						</span>
					</div>

					<div class="bars" role="img" aria-label={`The ${e.dir === 'f' ? 'forward' : 'backward'} expert's probability for each of the 20 amino acids at residue ${e.i + 1}`}>
						{#each [...w.alphabet] as a, j}
							{@const p = e.row[j]}
							<div class="col" class:hit={revealed && a === e.x}>
								<span class="stack">
									<span class="val">{p >= 0.15 || (revealed && a === e.x) ? p.toFixed(2) : ''}</span>
									<span class="bar" style:height={`${Math.max(0.5, 82 * p)}%`}></span>
								</span>
								<span class="aa">{a}</span>
							</div>
						{/each}
					</div>

					<p class="answer" aria-live="polite">
						{#if revealed}
							It is <span class="mono"><strong>{e.x}</strong></span>, given <strong>{e.q.toFixed(3)}</strong>:
							{e.rank === 1 ? 'its first choice' : `its ${ordinal(e.rank)} choice (first: ${e.ranked[0].a}, ${e.ranked[0].p.toFixed(2)})`}.
						{:else}
							<span class="muted">A probability for each amino acid, adding up to one.</span>
						{/if}
					</p>
				</div>
			{/each}
		</div>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.walk {
		--f: var(--sl-color-accent);
		--b: #ef785f;
		display: grid;
		gap: 0.6rem;
		font-size: var(--sl-text-sm);
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
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
		gap: 0.6rem;
	}
	.card {
		display: grid;
		gap: 0.45rem;
		align-content: start;
		padding: 0.5rem 0.65rem;
		border-radius: 0.45rem;
		border-left: 4px solid var(--c);
		background: color-mix(in srgb, var(--c) 8%, transparent);
	}
	.card.f {
		--c: var(--f);
	}
	.card.b {
		--c: var(--b);
	}
	.ask {
		display: grid;
		gap: 0.2rem;
		min-height: 4.2rem;
	}
	.who {
		font-weight: 700;
		color: var(--c);
	}
	.chips {
		display: inline-flex;
		gap: 2px;
		vertical-align: middle;
	}
	.chip {
		display: inline-grid;
		place-items: center;
		width: 1.2rem;
		height: 1.2rem;
		border-radius: 0.2rem;
		font-family: var(--sl-font-mono);
		font-size: var(--sl-text-xs);
		font-weight: 600;
		background: color-mix(in srgb, var(--c) 30%, transparent);
	}
	.chip.pad {
		background: transparent;
		border: 1px dashed var(--sl-color-gray-4);
		color: var(--sl-color-gray-3);
	}
	.bars {
		display: grid;
		grid-template-columns: repeat(20, 1fr);
		gap: 2px;
		height: 6.5rem;
	}
	.col {
		display: flex;
		flex-direction: column;
		align-items: center;
		height: 100%;
		min-width: 0;
	}
	.stack {
		flex: 1;
		min-height: 0;
		width: 100%;
		display: flex;
		flex-direction: column;
		justify-content: flex-end;
		align-items: center;
	}
	.bar {
		width: 100%;
		max-width: 1.2rem;
		border-radius: 2px 2px 0 0;
		background: var(--c);
		opacity: 0.75;
		transition: height 0.35s ease;
	}
	.col.hit .bar {
		opacity: 1;
		outline: 2px solid var(--sl-color-white);
	}
	.val {
		font-size: 9px;
		font-family: var(--sl-font-mono);
		color: var(--sl-color-gray-2);
		white-space: nowrap;
	}
	.aa {
		font-family: var(--sl-font-mono);
		font-size: 10px;
		color: var(--sl-color-gray-2);
	}
	.col.hit .aa {
		color: var(--sl-color-white);
		font-weight: 700;
	}
	.answer {
		margin: 0;
		min-height: 2.6rem;
		font-size: var(--sl-text-xs);
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
