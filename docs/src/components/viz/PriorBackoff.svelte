<script lang="ts">
	import { dataUrl, fmt } from './theme';
	import SequenceLens from './SequenceLens.svelte';

	interface Level {
		context: string;
		n: number;
		nx: number;
	}
	interface Prior {
		alpha: number[];
		probes: Record<string, string>;
		ladders: Record<string, { forward: Level[]; backward: Level[] }[]>;
		multipliers: number[];
		held_out: number[];
		n_train: number;
		n_test: number;
		unseen_share: number;
		reference_alpha: number[];
	}

	let d = $state<Prior | null>(null);
	let probe = $state('new');
	let pos = $state(7);
	let dir = $state<'f' | 'b'>('f');
	let step = $state(16); // index into the multipliers: 16 is x1, BEAM's own strengths
	let failed = $state(false);
	let width = $state(700);

	$effect(() => {
		fetch(dataUrl('viz/prior_strength/prior_strength.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((x: Prior) => (d = x))
			.catch(() => (failed = true));
	});

	const seq = $derived(d ? d.probes[probe] : '');
	const k = $derived(d ? d.alpha.length - 1 : 4);
	const mult = $derived(d ? d.multipliers[step] : 1);
	const x = $derived(seq[pos] ?? '');

	// The chain, from every residue equally likely up to the full context: at each level the
	// prediction is a weighted average of what the context was seen to do and what the level
	// below predicts, with weight n / (n + alpha) on the counts.
	const chain = $derived.by(() => {
		if (!d || !seq) return [];
		const levels = d.ladders[seq][pos][dir === 'f' ? 'forward' : 'backward'];
		let p = 1 / 20;
		return levels.map((l, j) => {
			const a = d!.alpha[j] * mult;
			const trust = l.n / (l.n + a);
			const own = l.n > 0 ? l.nx / l.n : null;
			const inherited = p;
			p = (l.nx + a * p) / (l.n + a);
			return { ...l, j, a, trust, own, inherited, p };
		});
	});

	// Layout of the ladder: one row per level, the full context on top, the uniform start at the
	// bottom; on the right a shared probability axis, on a log scale with a slot for zero.
	const rowH = 62;
	const top = 26;
	const L = $derived.by(() => {
		const textW = Math.min(300, Math.max(210, width * 0.42));
		const x0 = textW + 34; // the axis' 10^-5
		const x1 = width - 86; // the axis' 1
		const zero = textW + 14;
		return { textW, x0, x1, zero, height: top + (k + 2) * rowH + 8 };
	});
	const xs = (p: number | null) =>
		p === null || p <= 0 ? L.zero : L.x0 + ((Math.log10(Math.max(p, 1e-5)) + 5) / 5) * (L.x1 - L.x0);
	const rowY = (j: number) => top + (k - j) * rowH; // level j; the start is j = -1
	const ticks = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1];
	const tickLabel = (t: number) => (t >= 0.01 ? String(t) : `10⁻${'⁰¹²³⁴⁵⁶⁷⁸⁹'[-Math.log10(t)]}`);
	const prob = (p: number) => (p >= 0.001 ? p.toFixed(4) : p.toExponential(2));
	const strength = (a: number) => (a >= 1e5 ? `${(a / 1e6).toFixed(1)} million` : a >= 100 ? fmt(a) : a.toFixed(1));

	// Held out: how likely the new cells are, relative to the best strength.
	const best = $derived(d ? Math.max(...d.held_out) : 0);
	const curve = $derived.by(() => {
		if (!d) return null;
		const left = 46;
		const right = width - 16;
		const h = 150;
		const cx = (i: number) => left + (i / (d!.multipliers.length - 1)) * (right - left);
		const cy = (v: number) => 12 + (1 - Math.exp(v - best)) * (h - 40);
		const path = d.held_out.map((v, i) => `${i ? 'L' : 'M'}${cx(i)},${cy(v)}`).join('');
		return { left, right, h, cx, cy, path };
	});
	const ratio = $derived(d ? Math.exp(best - d.held_out[step]) : 1);
	const bestStep = $derived(d ? d.held_out.indexOf(best) : 0);

	function choose(name: string) {
		probe = name;
		pos = Math.min(pos, d!.probes[name].length - 1);
	}
	const label = (m: number) => (m < 0.1 ? m.toFixed(2) : m < 10 ? m.toFixed(2).replace(/\.?0+$/, '') : m.toFixed(0));
</script>

<div class="prior" bind:clientWidth={width}>
	{#if d && chain.length}
		{@const g = L}
		<div class="row" role="group" aria-label="Sequence">
			{#each Object.keys(d.probes) as name}
				<button class:on={probe === name} aria-pressed={probe === name} onclick={() => choose(name)}>
					{name}: <span class="mono">{d.probes[name]}</span>
				</button>
			{/each}
		</div>
		<div class="row" role="group" aria-label="Expert">
			<button class="f" class:on={dir === 'f'} aria-pressed={dir === 'f'} onclick={() => (dir = 'f')}>Forward expert →</button>
			<button class="b" class:on={dir === 'b'} aria-pressed={dir === 'b'} onclick={() => (dir = 'b')}>← Backward expert</button>
			<span class="muted small">Choose a residue below.</span>
		</div>
		<SequenceLens sequence={seq} {k} windows={[{ target: pos, dir }]} onpick={(i) => (pos = i)} />

		<svg width={width} height={g.height} role="img"
			aria-label={`The ${dir === 'f' ? 'forward' : 'backward'} expert's probability of ${x} at residue ${pos + 1}, level by level`}>
			<!-- the axis -->
			{#each ticks as t}
				<line x1={xs(t)} x2={xs(t)} y1={top - 6} y2={g.height - 8} class="grid" />
				<text x={xs(t)} y={top - 12} text-anchor="middle" class="tick">{tickLabel(t)}</text>
			{/each}
			<text x={g.zero} y={top - 12} text-anchor="middle" class="tick">0</text>
			<text x={g.x1 + 12} y={top - 12} class="tick">probability of {x}</text>

			<!-- the start: every residue equally likely -->
			<g transform={`translate(0, ${rowY(-1)})`}>
				<text x={0} y={22} class="lab">Every residue equally likely</text>
				<text x={0} y={40} class="sub">no information yet: 1/20</text>
				<path d={`M${xs(0.05)},${rowH / 2 - 7}l6,7l-6,7l-6,-7z`} class="res" />
				<text x={g.x1 + 12} y={rowH / 2 + 4} class="val">0.0500</text>
			</g>

			{#each chain as l}
				{@const yy = rowY(l.j)}
				{@const isTop = l.j === k}
				<!-- what the level below hands up -->
				<line x1={xs(l.inherited)} x2={xs(l.inherited)} y1={yy + rowH / 2} y2={yy + rowH + rowH / 2} class="hand" />
				<g transform={`translate(0, ${yy})`}>
					{#if isTop}<rect x={-6} y={2} width={width + 6} height={rowH - 4} rx="8" class="topbg" />{/if}
					<text x={0} y={18} class="lab">
						{l.j === 0 ? 'Position only' : `${l.j} residue${l.j > 1 ? 's' : ''} of context`}
					</text>
					{#if l.j > 0}
						{#each [...l.context] as c, ci}
							<rect x={(l.j === 1 ? 124 : 136) + ci * 15} y={6} width="13" height="15" rx="2" class="chip" class:pad={c === '·'} />
							<text x={(l.j === 1 ? 124 : 136) + ci * 15 + 6.5} y={18} text-anchor="middle" class="chipt">{c}</text>
						{/each}
					{/if}
					<text x={0} y={35} class="sub">
						{l.n === 0 ? 'this context was never seen' : `seen ${fmt(l.n)} time${l.n > 1 ? 's' : ''}, followed by ${x} ${fmt(l.nx)}`}
					</text>
					<text x={0} y={51} class="sub">
						weight on its counts <tspan class="strong">{(100 * l.trust).toFixed(l.trust > 0.995 || l.trust < 0.005 ? 1 : 0)}%</tspan> (α = {strength(l.a)})
					</text>

					<!-- the weighted average: own counts and inherited value, the result between them -->
					{#if l.own !== null}
						<line x1={xs(l.own)} x2={xs(l.inherited)} y1={rowH / 2} y2={rowH / 2} class="lever" />
						<circle cx={xs(l.own)} cy={rowH / 2} r={3 + 7 * l.trust} class="own"><title>{x} followed this context {fmt(l.nx)} of {fmt(l.n)} times: {prob(l.own)}</title></circle>
					{/if}
					<circle cx={xs(l.inherited)} cy={rowH / 2} r={3 + 7 * (1 - l.trust)} class="inh"><title>handed up from the level below: {prob(l.inherited)}</title></circle>
					<path d={`M${xs(l.p)},${rowH / 2 - 7}l6,7l-6,7l-6,-7z`} class="res" />
					<text x={g.x1 + 12} y={rowH / 2 + 4} class="val" class:answer={isTop}>{prob(l.p)}</text>
				</g>
			{/each}
		</svg>

		<div class="legend">
			<span><span class="dot own"></span>what this context was seen to do (its counts)</span>
			<span><span class="dot inh"></span>what the level below predicts</span>
			<span><span class="dia"></span>this level's answer, between the two</span>
			<span class="muted">dot size: the weight each gets</span>
		</div>

		<label class="slider">
			<span>Multiply every prior strength α by <strong>×{label(mult)}</strong>{step === 16 ? " (BEAM's own strengths)" : ''}</span>
			<input type="range" min="0" max={d.multipliers.length - 1} step="1" bind:value={step} aria-label="Multiply every prior strength" />
			<span class="ticks muted"><span>×0.01: trust the counts</span><span>×1</span><span>×100: trust the shorter contexts</span></span>
		</label>

		{#if curve}
			{@const c = curve}
			<div class="held">
				<p class="htitle">Does it predict new sequences? {fmt(d.n_test)} new cells from the same population, {Math.round(100 * d.unseen_share)}% of them sequences the model never saw</p>
				<svg width={width} height={c.h} role="img" aria-label="How likely the new cells are at each multiple of the prior strengths, relative to the best">
					{#each [0, 8, 16, 24, 32] as i}
						<line x1={c.cx(i)} x2={c.cx(i)} y1={10} y2={c.h - 28} class="grid" />
						<text x={c.cx(i)} y={c.h - 12} text-anchor="middle" class="tick">×{label(d.multipliers[i])}</text>
					{/each}
					{#each [1, 0.5, 0] as v}
						<text x={c.left - 6} y={12 + (1 - v) * (c.h - 40) + 4} text-anchor="end" class="tick">{v}</text>
					{/each}
					<line x1={c.cx(16)} x2={c.cx(16)} y1={10} y2={c.h - 28} class="beam" />
					<text x={c.cx(16) + 6} y={c.h - 34} class="beamlab">BEAM's strengths</text>
					<text x={c.left + 6} y={20} class="tick">a new cell's probability, relative to the best</text>
					<path d={c.path} class="curve" />
					<circle cx={c.cx(step)} cy={c.cy(d.held_out[step])} r="6" class="marker" />
				</svg>
				<p class="readout" aria-live="polite">
					{#if step === 16}
						At BEAM's own strengths, a new cell is on average {(1 / ratio).toFixed(2)} as likely as at the best multiple tried
						(×{label(d.multipliers[bestStep])}): the strengths chosen from the counts alone sit at the top of the curve.
					{:else if ratio < 1.005}
						At ×{label(mult)} a new cell is as likely as at the best multiple tried.
					{:else}
						At ×{label(mult)} a new cell is on average <strong>{ratio.toFixed(ratio < 2 ? 2 : 1)} times less likely</strong> than at the best
						strengths: {step < 16 ? 'trusting thin counts, the model gives too little to what it has not seen' : 'leaning on short contexts, the model forgets what longer ones show'}.
					{/if}
				</p>
			</div>
		{/if}
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.prior {
		--f: var(--sl-color-accent);
		--b: #ef785f;
		--inh: #6c7bd9;
		display: grid;
		gap: 0.6rem;
		font-size: var(--sl-text-sm);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: center;
	}
	button {
		font: inherit;
		font-size: var(--sl-text-xs);
		padding: 0.2rem 0.65rem;
		border-radius: 999px;
		border: 1px solid var(--sl-color-gray-5);
		background: transparent;
		color: var(--sl-color-white);
		cursor: pointer;
	}
	button.on {
		background: var(--sl-color-white);
		border-color: var(--sl-color-white);
		color: var(--sl-color-black);
	}
	button.f.on {
		background: var(--f);
		border-color: var(--f);
	}
	button.b.on {
		background: var(--b);
		border-color: var(--b);
	}
	svg {
		max-width: 100%;
		overflow: visible;
	}
	.grid {
		stroke: var(--sl-color-gray-5);
		stroke-dasharray: 2 3;
	}
	.tick {
		font-size: 10px;
		fill: var(--sl-color-gray-3);
	}
	.lab {
		font-size: 12.5px;
		font-weight: 600;
		fill: var(--sl-color-white);
	}
	.sub {
		font-size: 11px;
		fill: var(--sl-color-gray-2);
	}
	.strong {
		font-weight: 700;
		fill: var(--sl-color-white);
	}
	.chip {
		fill: color-mix(in srgb, var(--sl-color-accent) 30%, transparent);
	}
	.chip.pad {
		fill: transparent;
		stroke: var(--sl-color-gray-4);
		stroke-dasharray: 2 2;
	}
	.chipt {
		font-size: 10.5px;
		font-family: var(--sl-font-mono);
		font-weight: 600;
		fill: var(--sl-color-white);
	}
	.topbg {
		fill: color-mix(in srgb, var(--sl-color-white) 6%, transparent);
	}
	.hand {
		stroke: var(--inh);
		stroke-dasharray: 3 3;
		stroke-width: 1.5;
		transition: all 0.3s ease;
	}
	.lever {
		stroke: var(--sl-color-gray-3);
		stroke-width: 2;
		transition: all 0.3s ease;
	}
	.own {
		fill: var(--sl-color-accent);
		transition: all 0.3s ease;
	}
	.inh {
		fill: var(--inh);
		transition: all 0.3s ease;
	}
	.res {
		fill: var(--sl-color-white);
		stroke: var(--sl-color-black);
		stroke-width: 1;
		transition: all 0.3s ease;
	}
	.val {
		font-size: 12px;
		font-family: var(--sl-font-mono);
		fill: var(--sl-color-gray-2);
	}
	.val.answer {
		font-weight: 700;
		fill: var(--sl-color-white);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem 1rem;
		font-size: var(--sl-text-xs);
		color: var(--sl-color-gray-2);
	}
	.dot,
	.dia {
		display: inline-block;
		width: 0.7rem;
		height: 0.7rem;
		margin-inline-end: 0.3rem;
		vertical-align: -0.1rem;
	}
	.dot {
		border-radius: 50%;
	}
	.dot.own {
		background: var(--sl-color-accent);
	}
	.dot.inh {
		background: var(--inh);
	}
	.dia {
		background: var(--sl-color-white);
		transform: rotate(45deg) scale(0.8);
	}
	.slider {
		display: grid;
		gap: 0.2rem;
	}
	.slider input {
		width: 100%;
		accent-color: var(--inh);
	}
	.ticks {
		display: flex;
		justify-content: space-between;
		font-size: var(--sl-text-xs);
	}
	.held {
		display: grid;
		gap: 0.3rem;
		padding-top: 0.4rem;
		border-top: 1px solid var(--sl-color-gray-5);
	}
	.htitle {
		margin: 0;
		font-weight: 600;
	}
	.curve {
		fill: none;
		stroke: var(--sl-color-accent);
		stroke-width: 2.5;
	}
	.beam {
		stroke: var(--sl-color-white);
		stroke-dasharray: 4 3;
	}
	.beamlab {
		font-size: 10.5px;
		fill: var(--sl-color-white);
	}
	.marker {
		fill: var(--inh);
		stroke: var(--sl-color-white);
		stroke-width: 2;
		transition: all 0.3s ease;
	}
	.readout {
		margin: 0;
		font-size: var(--sl-text-xs);
		min-height: 2.4rem;
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.small {
		font-size: var(--sl-text-xs);
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
