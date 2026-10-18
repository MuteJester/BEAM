<script lang="ts">
	import { dataUrl } from './theme';

	let train = $state<string[]>([]);
	let k = $state(2);
	let t = $state(0);
	let playing = $state(false);
	let failed = $state(false);

	$effect(() => {
		fetch(dataUrl('viz/model_parts/model_parts.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d) => {
				train = d.toy.train;
				k = d.toy.k;
			})
			.catch(() => (failed = true));
	});

	// Every (sequence, residue) in order: one counting step each.
	const steps = $derived(train.flatMap((s, si) => [...s].map((_, i) => ({ si, i }))));
	const cur = $derived(steps[Math.min(t, steps.length - 1)]);
	const finished = $derived(steps.length > 0 && t >= steps.length - 1);

	$effect(() => {
		if (!playing || !steps.length) return;
		const timer = setInterval(() => {
			if (t >= steps.length - 1) playing = false;
			else t += 1;
		}, 450);
		return () => clearInterval(timer);
	});

	const pad = (ctx: string) => ctx.padStart(k, '·');
	// The row a residue is counted in, for each expert: its position from that expert's end and
	// the k residues before it in that expert's reading direction ('·' marks the boundary).
	function rowOf(s: string, i: number, backward: boolean) {
		const r = backward ? [...s].reverse().join('') : s;
		const j = backward ? s.length - 1 - i : i;
		return { position: j + 1, ctx: pad(r.slice(Math.max(0, j - k), j)), residue: r[j] };
	}
	function countsUpTo(backward: boolean) {
		const rows = new Map<string, Map<string, number>>();
		for (let n = 0; n <= t && n < steps.length; n++) {
			const { si, i } = steps[n];
			const row = rowOf(train[si], i, backward);
			const key = `${row.position}|${row.ctx}`;
			if (!rows.has(key)) rows.set(key, new Map());
			const m = rows.get(key)!;
			m.set(row.residue, (m.get(row.residue) ?? 0) + 1);
		}
		return rows;
	}
	const fwdCounts = $derived(train.length ? countsUpTo(false) : new Map());
	const bwdCounts = $derived(train.length ? countsUpTo(true) : new Map());

	function card(backward: boolean) {
		if (!cur) return null;
		const row = rowOf(train[cur.si], cur.i, backward);
		const m = (backward ? bwdCounts : fwdCounts).get(`${row.position}|${row.ctx}`) ?? new Map();
		const cells = [...m.entries()].sort((a, b) => b[1] - a[1]);
		const total = cells.reduce((s, [, n]) => s + n, 0);
		return { ...row, cells, total, rows: (backward ? bwdCounts : fwdCounts).size };
	}
	const fwd = $derived(card(false));
	const bwd = $derived(card(true));

	function toggle() {
		if (finished) t = 0;
		playing = finished || !playing;
	}
</script>

<div class="counting">
	{#if steps.length && cur && fwd && bwd}
		<div class="controls">
			<button class="primary" onclick={toggle}>{playing ? 'Pause' : finished ? 'Count again' : t ? 'Resume' : 'Count'}</button>
			<button onclick={() => ((playing = false), (t = Math.min(steps.length - 1, t + 1)))}>Step ▶</button>
			<button onclick={() => ((playing = false), (t = 0))}>Reset</button>
			<span class="muted">sequence {cur.si + 1} of {train.length}, residue {cur.i + 1}</span>
		</div>

		<div class="seqs" aria-label="Training sequences">
			{#each train as s, si}
				<div class="seq" class:done={si < cur.si} class:now={si === cur.si}>
					{#each [...s] as ch, i}
						{@const fwdCtx = si === cur.si && i < cur.i && i >= cur.i - k}
						{@const bwdCtx = si === cur.si && i > cur.i && i <= cur.i + k}
						<span class="res" class:here={si === cur.si && i === cur.i} class:fctx={fwdCtx} class:bctx={bwdCtx}>{ch}</span>
					{/each}
				</div>
			{/each}
		</div>

		<div class="cards">
			{#each [{ c: fwd, name: 'Forward expert', side: 'from the start', ctxWord: 'after', cls: 'f' }, { c: bwd, name: 'Backward expert', side: 'from the end (reading backwards)', ctxWord: 'after', cls: 'b' }] as e}
				<div class="card {e.cls}">
					<div class="title">{e.name}</div>
					<div class="key">
						row: position <strong>{e.c.position}</strong> {e.side}, {e.ctxWord}
						<span class="mono ctx">{e.c.ctx}</span>
					</div>
					<div class="chips" aria-live="polite">
						{#each e.c.cells as [res, n]}
							<span class="chip" class:inc={res === e.c.residue}><span class="mono">{res}</span> × {n}</span>
						{/each}
					</div>
					<div class="muted small">
						{e.c.residue} counted here: {e.c.cells.find(([r]) => r === e.c.residue)?.[1] ?? 0} of {e.c.total} in this row;
						{e.c.rows} rows filled so far
					</div>
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
	.counting {
		display: grid;
		gap: 0.7rem;
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
		background: var(--sl-color-accent);
		border-color: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	.seqs {
		display: grid;
		gap: 0.25rem;
	}
	.seq {
		display: flex;
		gap: 2px;
		opacity: 0.45;
	}
	.seq.done {
		opacity: 0.7;
	}
	.seq.now {
		opacity: 1;
	}
	.res {
		display: inline-grid;
		place-items: center;
		width: 1.45rem;
		height: 1.45rem;
		font-family: var(--sl-font-mono);
		font-size: var(--sl-text-xs);
		border-radius: 0.25rem;
		border: 2px solid transparent;
		background: var(--sl-color-gray-6);
	}
	.res.here {
		background: var(--sl-color-white);
		color: var(--sl-color-black);
		font-weight: 700;
	}
	.res.fctx {
		border-color: var(--sl-color-text-accent);
	}
	.res.bctx {
		border-color: #ef785f;
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
		gap: 0.6rem;
	}
	.card {
		border: 1px solid var(--sl-color-gray-5);
		border-radius: 0.45rem;
		padding: 0.55rem 0.7rem;
		display: grid;
		gap: 0.35rem;
	}
	.card.f {
		border-left: 4px solid var(--sl-color-text-accent);
	}
	.card.b {
		border-left: 4px solid #ef785f;
	}
	.title {
		font-weight: 600;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		min-height: 1.6rem;
	}
	.chip {
		padding: 0.1rem 0.5rem;
		border-radius: 999px;
		background: var(--sl-color-gray-6);
		font-size: var(--sl-text-xs);
	}
	.chip.inc {
		background: var(--sl-color-accent);
		color: var(--sl-color-black);
		font-weight: 600;
	}
	.card.b .chip.inc {
		background: #ef785f;
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.ctx {
		padding: 0 0.3rem;
		border-radius: 0.2rem;
		background: var(--sl-color-gray-6);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
	.small {
		font-size: var(--sl-text-xs);
	}
</style>
