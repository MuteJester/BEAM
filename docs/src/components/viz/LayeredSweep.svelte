<script lang="ts">
	import { dataUrl, fmt, sci } from './theme';

	interface Layer {
		states: string[];
		W: number[];
	}
	interface Sweep {
		alphabet: string;
		k: number;
		length: number;
		layers: Layer[];
		edges: [number, number, number][][]; // per position: [from state, to state, factor]
		terminal: number[];
		Z: number;
		listed: number;
		n_sequences: number;
		reference: { length: number; n_nodes: number; n_edges: number };
	}

	let sw = $state<Sweep | null>(null);
	let layer = $state(0); // positions swept so far; length + 1 adds the end
	let playing = $state(false);
	let sel = $state<{ j: number; i: number } | null>(null);
	let failed = $state(false);
	let width = $state(640);

	$effect(() => {
		fetch(dataUrl('viz/sweep_toy/sweep_toy.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: Sweep) => (sw = d))
			.catch(() => (failed = true));
	});

	const L = $derived(sw ? sw.length : 0);
	const done = $derived(sw !== null && layer > L);

	$effect(() => {
		if (!playing || !sw) return;
		const timer = setInterval(() => {
			if (layer > L) playing = false;
			else layer += 1;
		}, 900);
		return () => clearInterval(timer);
	});

	function toggle() {
		if (done) layer = 0;
		sel = null;
		playing = done || !playing;
	}

	// Layout: one column per position, the start on the left and the end on the right. A state is
	// the last two residues written; rows are grouped by the residue written last.
	const rowH = 12;
	const blockGap = 5;
	const top = 30;
	const labelW = 30;
	const geom = $derived.by(() => {
		const cols = L + 2;
		const colW = Math.max(30, Math.min(72, (width - labelW - 16) / cols));
		const x = (j: number) => labelW + 12 + j * colW;
		const height = top + 5 * (5 * rowH + blockGap);
		const mid = top + (height - top) / 2 - blockGap / 2;
		return { cols, colW, x, height, mid };
	});
	function y(st: string, j: number) {
		if (!sw || j === 0) return geom.mid;
		const b = st[st.length - 1];
		const a = st[st.length - 2];
		const r = a === '·' ? 2 : sw.alphabet.indexOf(a);
		return top + sw.alphabet.indexOf(b) * (5 * rowH + blockGap) + r * rowH + rowH / 2;
	}
	const rowLabels = $derived(
		sw ? [...sw.alphabet].flatMap((b) => [...sw!.alphabet].map((a) => ({ st: a + b, y: y(a + b, 2) }))) : [],
	);

	const totals = $derived(sw ? sw.layers.map((l) => l.W.reduce((s, w) => s + w, 0)) : []);
	// Every edge, with its share of the weight arriving in its column.
	const paths = $derived.by(() => {
		if (!sw) return [];
		const out: { j: number; from: number; to: number; d: string; share: number; factor: number }[] = [];
		sw.edges.forEach((E, jj) => {
			const j = jj + 1;
			const prev = sw!.layers[j - 1];
			const next = sw!.layers[j];
			for (const [from, to, factor] of E) {
				const d = `M${geom.x(j - 1)},${y(prev.states[from], j - 1)}L${geom.x(j)},${y(next.states[to], j)}`;
				out.push({ j, from, to, d, share: (prev.W[from] * factor) / totals[j], factor });
			}
		});
		const last = sw.layers[L];
		last.states.forEach((st, i) => {
			const d = `M${geom.x(L)},${y(st, L)}L${geom.x(L + 1)},${geom.mid}`;
			out.push({ j: L + 1, from: i, to: 0, d, share: (last.W[i] * sw!.terminal[i]) / sw!.Z, factor: sw!.terminal[i] });
		});
		return out;
	});
	const radius = (j: number, i: number) => 1.6 + 7 * Math.sqrt(sw!.layers[j].W[i] / totals[j]);

	const edgesWalked = $derived(sw ? paths.filter((p) => p.j <= layer).length : 0);
	const partial = (j: number) => Math.min(sw!.n_sequences, sw!.alphabet.length ** j);

	// The selected state's total, term by term: one term per edge into it.
	const terms = $derived.by(() => {
		if (!sw || !sel) return null;
		const s = sel;
		const into = paths.filter((p) => p.j === s.j && p.to === s.i);
		const prev = s.j <= L ? sw.layers[s.j - 1] : sw.layers[L];
		return into
			.map((p) => ({ st: prev.states[p.from], w: prev.W[p.from], f: p.factor }))
			.sort((u, v) => v.w * v.f - u.w * u.f);
	});
	const selName = $derived(sw && sel ? (sel.j > L ? `Z(${L})` : `W(${sw.layers[sel.j].states[sel.i]})`) : '');
	const selValue = $derived(sw && sel ? (sel.j > L ? sw.Z : sw.layers[sel.j].W[sel.i]) : 0);
	const num = (v: number) => (v >= 0.001 ? v.toFixed(4) : sci(v, 3));

	function pick(j: number, i: number) {
		if (j === 0 || j > layer) return;
		playing = false;
		sel = sel && sel.j === j && sel.i === i ? null : { j, i };
	}
	// An edge is as visible as the share of weight it carries, so the main flows stand out.
	function opacity(p: { j: number; to: number; share: number }) {
		if (sel) return p.j === sel.j && p.to === sel.i ? 1 : 0.04;
		if (p.j > layer) return 0.05;
		const strength = Math.min(1, 1.6 * Math.sqrt(p.share));
		return (p.j === layer ? 0.12 : 0.05) + 0.85 * strength;
	}
	const logW = (v: number) => (Math.log10(Math.max(1, v)) / Math.log10(sw!.n_sequences)) * 100;
</script>

<div class="sweep" bind:clientWidth={width}>
	{#if sw}
		{@const g = geom}
		<div class="controls">
			<button class="primary" onclick={toggle}>{playing ? 'Pause' : done ? 'Sweep again' : layer ? 'Resume' : 'Sweep'}</button>
			<button onclick={() => ((playing = false), (sel = null), (layer = Math.min(L + 1, layer + 1)))}>Step ▶︎</button>
			<button onclick={() => ((playing = false), (sel = null), (layer = 0))}>Reset</button>
			<span class="muted">{layer === 0 ? 'ready' : layer <= L ? `position ${layer} of ${L}` : 'the end: Z'}</span>
		</div>

		<svg width={g.x(L + 1) + 30} height={g.height + 6} role="group"
			aria-label={`The layered graph of a small DNA model for length ${L}: one column per position, one dot per state, swept from left to right`}>
			{#each Array(L + 2) as _, j}
				<text x={g.x(j)} y={12} text-anchor="middle" class="head" class:on={j <= layer}>{j === 0 ? 'start' : j > L ? 'end' : j}</text>
			{/each}
			{#each rowLabels as r}
				<text x={labelW} y={r.y + 3} text-anchor="end" class="rowlab">{r.st}</text>
			{/each}

			{#each paths as p}
				<path
					d={p.d}
					class="edge"
					class:lit={p.j < layer}
					class:front={p.j === layer}
					class:into={sel !== null && p.j === sel.j && p.to === sel.i}
					style:stroke-width={p.j <= layer ? 0.5 + 7 * p.share : 0.4}
					style:opacity={opacity(p)}
				/>
			{/each}

			{#each sw.layers as l, j}
				{#each l.states as st, i}
					<circle
						cx={g.x(j)}
						cy={y(st, j)}
						r={j <= layer ? radius(j, i) : 1.4}
						class="node"
						class:lit={j <= layer}
						class:front={j === layer}
						class:picked={sel !== null && sel.j === j && sel.i === i}
						class:source={sel !== null && sel.j === j + 1 && terms?.some((t) => t.st === st)}
						role="button"
						tabindex={j > 0 && j <= layer ? 0 : -1}
						aria-label={`${st} at position ${j}`}
						onclick={() => pick(j, i)}
						onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && pick(j, i)}
					><title>{st} after position {j}: total {num(l.W[i])}</title></circle>
				{/each}
			{/each}
			<circle
				cx={g.x(L + 1)}
				cy={g.mid}
				r={done ? 9 : 1.4}
				class="node end"
				class:lit={done}
				class:picked={sel !== null && sel.j > L}
				role="button"
				tabindex={done ? 0 : -1}
				aria-label="The end: Z"
				onclick={() => pick(L + 1, 0)}
				onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && pick(L + 1, 0)}
			/>
			{#if done}<text x={g.x(L + 1)} y={g.mid - 14} text-anchor="middle" class="zlab">Z({L})</text>{/if}
		</svg>

		<div class="compare" aria-label="Sequences summed against edges walked">
			<span class="clab muted">on a log scale</span><span></span><span></span>
			<span class="clab">sequences summed</span>
			<span class="track"><span class="cbar list" style:width={`${layer ? logW(partial(Math.min(layer, L))) : 0}%`}></span></span>
			<span class="cnum mono">{layer ? fmt(partial(Math.min(layer, L))) : 0}</span>
			<span class="clab">edges walked</span>
			<span class="track"><span class="cbar walk" style:width={`${logW(edgesWalked)}%`}></span></span>
			<span class="cnum mono">{fmt(edgesWalked)}</span>
		</div>

		<div class="readout" aria-live="polite">
			{#if sel && terms}
				<span class="mono">
					{selName} = {#each terms.slice(0, 5) as t, n}{n ? ' + ' : ''}{num(t.w)} × {num(t.f)}{/each}{terms.length > 5
						? ` + … (${terms.length - 5} smaller terms)`
						: ''} = <strong>{num(selValue)}</strong>
				</span>
				<span>
					{#if sel.j > L}
						Z({L}) collects every state of the last position, each times the factors still owed to its last two residues.
					{:else}
						The total of state <span class="mono">{sw.layers[sel.j].states[sel.i]}</span> after position {sel.j}: one term for each
						state it can be reached from, that state's total times the factor of the step. It stands for all
						{fmt(partial(sel.j) / sw.layers[sel.j].states.length)} partial sequences ending in these two residues.
					{/if}
				</span>
			{:else if layer === 0}
				<span class="muted">Each column is one position of a DNA sequence of length {L}; each dot is a state, the last two residues
					written. Press Sweep.</span>
			{:else if layer <= L}
				<span>
					Position {layer}: all {fmt(partial(layer))} possible first {layer} residue{layer > 1 ? 's' : ''}, summed into
					{sw.layers[layer].states.length} state{sw.layers[layer].states.length > 1 ? 's' : ''}. Each state adds up the ways into it; dot
					size is its share of the column's total. Choose a dot to see its sum.
				</span>
			{:else}
				<span>
					Z({L}) = <strong>{sw.Z.toFixed(6)}</strong>, the sum over all {fmt(sw.n_sequences)} sequences of length {L}, from
					{fmt(edgesWalked)} edges. Listing every sequence and adding them up gives the same: {sw.listed.toFixed(6)}.
				</span>
				<span class="muted">
					The reference model's graph for length {sw.reference.length} has {fmt(sw.reference.n_nodes)} states and
					{fmt(sw.reference.n_edges)} edges, and covers {sci(20 ** sw.reference.length, 2)} sequences the same way.
				</span>
			{/if}
		</div>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.sweep {
		display: grid;
		gap: 0.55rem;
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
	svg {
		max-width: 100%;
		overflow: visible;
	}
	.head {
		font-size: 11px;
		fill: var(--sl-color-gray-4);
	}
	.head.on {
		fill: var(--sl-color-gray-2);
		font-weight: 600;
	}
	.rowlab {
		font-size: 8.5px;
		font-family: var(--sl-font-mono);
		fill: var(--sl-color-gray-3);
	}
	.edge {
		stroke: var(--sl-color-gray-4);
		stroke-linecap: round;
		transition: opacity 0.3s ease, stroke-width 0.4s ease;
	}
	.edge.lit {
		stroke: #8b97e6;
	}
	.edge.front {
		stroke: var(--sl-color-white);
	}
	.edge.into {
		stroke: #ef785f;
	}
	.node {
		fill: var(--sl-color-gray-5);
		transition: r 0.4s ease, fill 0.3s ease;
	}
	.node.lit {
		fill: var(--sl-color-accent);
		cursor: pointer;
	}
	.node.front {
		fill: var(--sl-color-white);
	}
	.node.source {
		fill: #ef785f;
	}
	.node.picked {
		stroke: #ef785f;
		stroke-width: 2.5;
	}
	.node.end.lit {
		fill: var(--sl-color-white);
	}
	.zlab {
		font-size: 11px;
		font-weight: 700;
		fill: var(--sl-color-white);
	}
	.compare {
		display: grid;
		grid-template-columns: auto 1fr auto;
		gap: 0.25rem 0.6rem;
		align-items: center;
		font-size: var(--sl-text-xs);
	}
	.clab {
		color: var(--sl-color-gray-2);
	}
	.track {
		height: 0.6rem;
		border-radius: 999px;
		background: var(--sl-color-gray-6);
		overflow: hidden;
	}
	.cbar {
		display: block;
		height: 100%;
		border-radius: 999px;
		transition: width 0.4s ease;
	}
	.cbar.list {
		background: var(--sl-color-gray-3);
	}
	.cbar.walk {
		background: var(--sl-color-accent);
	}
	.cnum {
		text-align: right;
		min-width: 4.5rem;
	}
	.readout {
		display: grid;
		gap: 0.3rem;
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
