<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, onThemeChange, pct, themeColors } from './theme';

	interface Length {
		length: number;
		p: number;
		q: number;
		length_term: number;
		pattern_term: number;
		windows: number;
	}
	interface Window {
		length: number;
		start: number;
		contribution: number;
	}
	interface Word {
		length: number;
		start: number;
		word: string;
		p: number;
		q: number;
		contribution: number;
	}
	interface Data {
		distance: number;
		squared: number;
		length_part: number;
		pattern_part: number;
		reported: number;
		remainder: number;
		width: number;
		lengths: Length[];
		windows: Window[];
		words: Word[];
	}

	let data = $state<Data | null>(null);
	let step = $state(1);
	let sel = $state<{ length: number; start: number } | null>(null);
	let themeTick = $state(0);
	let failed = $state(false);
	let lengthChart: HTMLDivElement | undefined = $state();
	let wordChart: HTMLDivElement | undefined = $state();
	let totalChart: HTMLDivElement | undefined = $state();
	let gridWidth = $state(600);

	const B_COLOUR = '#6c7bd9';
	const steps = ['Lengths', 'Windows', 'Words', 'Total'];

	$effect(() => {
		fetch(dataUrl('viz/distance_anatomy/distance_anatomy.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: Data) => {
				data = d;
				const top = [...d.windows].sort((a, b) => b.contribution - a.contribution)[0];
				sel = top ? { length: top.length, start: top.start } : null;
			})
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	const f4 = (x: number) => x.toFixed(4);
	const f6 = (x: number) => x.toFixed(6);
	const share = (x: number) => (data ? pct(x / data.squared) : '');

	const selLength = $derived(data && sel ? data.lengths.find((l) => l.length === sel!.length) : undefined);
	const selWindow = $derived(
		data && sel ? data.windows.find((w) => w.length === sel!.length && w.start === sel!.start) : undefined,
	);
	const selWords = $derived(
		data && sel ? data.words.filter((w) => w.length === sel!.length && w.start === sel!.start) : [],
	);
	const topLengths = $derived(
		data ? [...data.lengths].sort((a, b) => b.length_term - a.length_term).slice(0, 3) : [],
	);

	// The window grid: one row per length, one cell per window start.
	const grid = $derived.by(() => {
		if (!data) return null;
		const lengths = data.lengths.map((l) => l.length).sort((a, b) => a - b);
		const maxStart = Math.max(...data.windows.map((w) => w.start));
		const positive = data.windows.filter((w) => w.contribution > 0).map((w) => w.contribution);
		const colour = Plot.scale({
			color: { type: 'log', scheme: 'viridis', domain: [Math.min(...positive), Math.max(...positive)] },
		});
		return { lengths, maxStart, colour };
	});

	$effect(() => {
		themeTick;
		if (!data || step !== 1 || !lengthChart) return;
		const c = themeColors();
		const rows = data.lengths;
		const bars = rows.flatMap((l) => [
			{ x1: l.length - 0.42, x2: l.length, y: l.p, who: 'A' },
			{ x1: l.length, x2: l.length + 0.42, y: l.q, who: 'B' },
		]);
		const w = lengthChart.clientWidth;
		const top = Plot.plot({
			width: w,
			height: 170,
			marginLeft: 56,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { label: null, tickFormat: 'd' },
			y: { label: '↑ π(L)', grid: true },
			marks: [
				Plot.rectY(bars, { x1: 'x1', x2: 'x2', y: 'y', fill: (d) => (d.who === 'A' ? c.accent : B_COLOUR) }),
				Plot.ruleY([0], { stroke: c.muted }),
			],
		});
		const bottom = Plot.plot({
			width: w,
			height: 130,
			marginLeft: 56,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { label: 'Sequence length L →', tickFormat: 'd' },
			y: { label: '↑ length term', grid: true },
			marks: [
				Plot.rectY(rows, { x1: (l) => l.length - 0.42, x2: (l) => l.length + 0.42, y: 'length_term', fill: c.second }),
				Plot.ruleY([0], { stroke: c.muted }),
			],
		});
		lengthChart.replaceChildren(top, bottom);
	});

	$effect(() => {
		themeTick;
		if (!data || step !== 3 || !wordChart || !selWords.length) return;
		const c = themeColors();
		const rows = selWords.flatMap((w) => [
			{ word: w.word, who: 'A', v: w.p },
			{ word: w.word, who: 'B', v: w.q },
		]);
		const plot = Plot.plot({
			width: wordChart.clientWidth,
			height: 34 + selWords.length * 30,
			marginLeft: 70,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { label: 'Probability given the length →', grid: true },
			y: { domain: selWords.map((w) => w.word), label: null },
			marks: [
				// A on the upper half of each word's row, B on the lower half.
				Plot.barX(rows.filter((d) => d.who === 'A'), { x: 'v', y: 'word', fill: c.accent, insetTop: 3, insetBottom: 14 }),
				Plot.barX(rows.filter((d) => d.who === 'B'), { x: 'v', y: 'word', fill: B_COLOUR, insetTop: 14, insetBottom: 3 }),
				Plot.ruleX([0], { stroke: c.muted }),
			],
		});
		wordChart.replaceChildren(plot);
	});

	$effect(() => {
		themeTick;
		if (!data || step !== 4 || !totalChart) return;
		const c = themeColors();
		const rows = data.lengths.flatMap((l) => [
			{ length: l.length, part: 'length', v: l.length_term },
			{ length: l.length, part: 'pattern', v: l.pattern_term },
		]);
		const plot = Plot.plot({
			width: totalChart.clientWidth,
			height: 220,
			marginLeft: 56,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { label: 'Sequence length L →', tickFormat: 'd' },
			y: { label: '↑ contribution to D²', grid: true },
			marks: [
				Plot.rectY(
					rows,
					Plot.stackY({
						x1: (d) => d.length - 0.42,
						x2: (d) => d.length + 0.42,
						y: 'v',
						fill: (d) => (d.part === 'length' ? c.second : c.accent),
						order: ['length', 'pattern'],
						z: 'part',
					}),
				),
				Plot.ruleY([0], { stroke: c.muted }),
			],
		});
		totalChart.replaceChildren(plot);
	});

	function pick(length: number, start: number) {
		sel = { length, start };
	}
</script>

<div class="anatomy" bind:clientWidth={gridWidth}>
	{#if data && grid}
		<!-- The running total: where each step sits in D². -->
		<div class="ledger" aria-label="The squared distance as its two parts">
			<div class="bar">
				<span
					class="seg length"
					class:dim={step === 2 || step === 3}
					style:width={pct(data.length_part / data.squared)}
					title="length part"
				></span>
				<span
					class="seg pattern"
					class:dim={step === 1}
					style:width={pct(data.pattern_part / data.squared)}
					title="pattern part"
				></span>
			</div>
			<div class="legend">
				<span><span class="swatch length"></span>length part {f4(data.length_part)} ({share(data.length_part)})</span>
				<span><span class="swatch pattern"></span>pattern part {f4(data.pattern_part)} ({share(data.pattern_part)})</span>
				<span>D² = {f4(data.squared)}, D = {f4(data.distance)}</span>
			</div>
		</div>

		<div class="steps" role="group" aria-label="Steps">
			{#each steps as name, i}
				<button class:active={step === i + 1} aria-pressed={step === i + 1} onclick={() => (step = i + 1)}>
					<span class="num">{i + 1}</span>{name}
				</button>
			{/each}
		</div>

		{#if step === 1}
			<p>
				<strong>Lengths.</strong> Each repertoire has a length distribution π(L): <span class="a">repertoire A</span>
				and <span class="b">repertoire B</span> above. Each length adds
				<span class="math">½ (√π<sub>A</sub>(L) − √π<sub>B</sub>(L))²</span> to D², drawn below; together these make the
				length part, {f4(data.length_part)}. The largest come from lengths
				{topLengths.map((l) => l.length).join(', ')}.
			</p>
			<div bind:this={lengthChart} role="img" aria-label="The two length distributions, and each length's term in the length part"></div>
			<pre class="code"><code>[(x.length, x.p, x.q, x.contribution) for x in r.lengths]
sum(x.contribution for x in r.lengths) == r.length_squared_distance</code></pre>
		{:else if step === 2}
			{@const cols = grid.maxStart + 1}
			{@const left = 40}
			{@const cw = Math.max(8, Math.min(26, (gridWidth - left - 8) / cols))}
			{@const ch = 14}
			<p>
				<strong>Windows.</strong> Within each length, BEAM compares the two models' probabilities of every short window of
				{data.width} residues, at every position. Each cell is one window's share of the pattern part:
				<span class="math">√(π<sub>A</sub>π<sub>B</sub>) · h²(window) / V<sub>L</sub></span>, where h² is the squared
				Hellinger distance between the two models' word probabilities in that window and V<sub>L</sub> the number of
				windows at length L. Choose a cell to see its words.
			</p>
			<svg
				class="grid"
				width={left + cols * cw + 4}
				height={grid.lengths.length * ch + 26}
				role="group"
				aria-label="Window contributions by length and position"
			>
				{#each grid.lengths as L, row}
					<text x={left - 6} y={row * ch + ch - 3} text-anchor="end" class="tick">{L}</text>
				{/each}
				{#each data.windows as w}
					{@const row = grid.lengths.indexOf(w.length)}
					<rect
						x={left + w.start * cw}
						y={row * ch}
						width={cw - 1}
						height={ch - 1}
						fill={w.contribution > 0 ? grid.colour.apply(w.contribution) : 'transparent'}
						class:selected={sel?.length === w.length && sel?.start === w.start}
						role="button"
						tabindex="0"
						aria-label={`length ${w.length}, window at ${w.start}: ${f6(w.contribution)}`}
						onclick={() => pick(w.length, w.start)}
						onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && pick(w.length, w.start)}
					><title>length {w.length}, positions {w.start}–{w.start + data.width - 1}: {f6(w.contribution)}</title></rect>
				{/each}
				<text x={left} y={grid.lengths.length * ch + 18} class="tick">window start → (rows: sequence length)</text>
			</svg>
			{#if selWindow && selLength}
				<p class="readout" aria-live="polite">
					Length {selWindow.length}, positions {selWindow.start}–{selWindow.start + data.width - 1}: {f6(selWindow.contribution)},
					{pct(selWindow.contribution / selLength.pattern_term)} of length {selLength.length}'s pattern term
					({f6(selLength.pattern_term)}). <button class="link" onclick={() => (step = 3)}>Show its words →</button>
				</p>
			{/if}
			<pre class="code"><code>[(w.length, w.start, w.contribution) for w in r.windows]
sum(w.contribution for w in r.windows if w.length == L) == x.pattern_contribution  # x: length L in r.lengths</code></pre>
		{:else if step === 3}
			{#if selWindow && selLength && selWords.length}
				{@const w = selWords[0]}
				{@const V = selLength.windows}
				{@const listed = selWords.reduce((s, x) => s + x.contribution, 0)}
				<p>
					<strong>Words.</strong> In the window at positions {selWindow.start}–{selWindow.start + data.width - 1} of
					length-{selLength.length} sequences, these words differ most between the two models (probability given the
					length, <span class="a">A</span> and <span class="b">B</span>). Each word adds
					<span class="math">√(π<sub>A</sub>π<sub>B</sub>) · (√p − √q)² / (2 V<sub>L</sub>)</span>; for {w.word}:
				</p>
				<p class="math worked">
					√({selLength.p.toFixed(4)} × {selLength.q.toFixed(4)}) × (√{w.p.toFixed(4)} − √{w.q.toFixed(4)})² / (2 × {V})
					= {f6(w.contribution)}
				</p>
				<div bind:this={wordChart} role="img" aria-label="The window's words that differ most, with their probability in each repertoire"></div>
				<p class="readout" aria-live="polite">
					The {selWords.length} words shown carry {pct(listed / selWindow.contribution)} of this window's
					{f6(selWindow.contribution)}; the rest is spread over many words, each adding very little.
				</p>
				<pre class="code"><code>[(p.word, p.conditional_p, p.conditional_q, p.contribution)
 for p in r.patterns if (p.length, p.start) == ({selWindow.length}, {selWindow.start})]
r.group_overlaps()   # overlapping words joined into longer motifs</code></pre>
			{:else}
				<p class="muted">Choose a window in step 2 to see its words.</p>
			{/if}
		{:else}
			<p>
				<strong>Total.</strong> Every term is non-negative and they add up: length part {f4(data.length_part)} +
				pattern part {f4(data.pattern_part)} = D² = {f4(data.squared)}, so D = √{f4(data.squared)} =
				<strong>{f4(data.distance)}</strong>. Each length's bar below is its length term and its pattern term, so the
				chart shows at a glance which lengths make the two repertoires differ, and how.
			</p>
			<div bind:this={totalChart} role="img" aria-label="Each length's length term and pattern term, stacked"></div>
			<div class="legend">
				<span><span class="swatch length"></span>length term</span>
				<span><span class="swatch pattern"></span>pattern term</span>
			</div>
			<pre class="code"><code>r.squared_distance == r.length_squared_distance + r.pattern_squared_distance
r.distance           # its square root</code></pre>
		{/if}
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.anatomy {
		display: grid;
		gap: 0.7rem;
		font-size: var(--sl-text-sm);
	}
	.anatomy p {
		margin: 0;
	}
	.ledger .bar {
		display: flex;
		height: 0.9rem;
		border-radius: 0.3rem;
		overflow: hidden;
		background: var(--sl-color-gray-6);
	}
	.seg {
		display: block;
		height: 100%;
		transition: opacity 0.2s ease;
	}
	.seg.length,
	.swatch.length {
		background: #ef785f;
	}
	.seg.pattern,
	.swatch.pattern {
		background: var(--sl-color-accent);
	}
	.seg.dim {
		opacity: 0.25;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem 1.1rem;
		margin-top: 0.35rem;
		font-size: var(--sl-text-xs);
		color: var(--sl-color-gray-2);
	}
	.swatch {
		display: inline-block;
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 2px;
		margin-inline-end: 0.35rem;
		vertical-align: -0.05rem;
	}
	.steps {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
	}
	.steps button {
		font: inherit;
		font-size: var(--sl-text-xs);
		padding: 0.25rem 0.75rem 0.25rem 0.3rem;
		border-radius: 999px;
		border: 1px solid var(--sl-color-gray-5);
		background: transparent;
		color: var(--sl-color-white);
		cursor: pointer;
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
	}
	.steps .num {
		display: inline-grid;
		place-items: center;
		width: 1.25rem;
		height: 1.25rem;
		border-radius: 50%;
		background: var(--sl-color-gray-6);
		font-weight: 600;
	}
	.steps button.active {
		border-color: var(--sl-color-accent);
	}
	.steps button.active .num {
		background: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	.a {
		color: var(--sl-color-text-accent);
		font-weight: 600;
	}
	.b {
		color: #6c7bd9;
		font-weight: 600;
	}
	:global(:root[data-theme='light']) .b {
		color: #4b5cc4;
	}
	.math {
		font-family: var(--sl-font-mono);
		font-size: 0.95em;
		white-space: nowrap;
	}
	.worked {
		white-space: normal;
		padding: 0.4rem 0.6rem;
		border-left: 3px solid var(--sl-color-accent);
		background: var(--sl-color-gray-7, var(--sl-color-black));
	}
	.grid {
		max-width: 100%;
		overflow: visible;
	}
	.grid rect {
		cursor: pointer;
		stroke: transparent;
		stroke-width: 2;
	}
	.grid rect:hover,
	.grid rect:focus-visible {
		stroke: var(--sl-color-white);
		outline: none;
	}
	.grid rect.selected {
		stroke: var(--sl-color-white);
	}
	.tick {
		font-size: 10px;
		fill: var(--sl-color-gray-3);
	}
	.code {
		margin: 0;
		padding: 0.5rem 0.7rem;
		font-size: var(--sl-text-xs);
		overflow-x: auto;
		border-radius: 0.3rem;
		background: var(--sl-color-gray-7, var(--sl-color-black));
	}
	.readout {
		font-size: var(--sl-text-sm);
	}
	.link {
		font: inherit;
		background: none;
		border: none;
		padding: 0;
		color: var(--sl-color-text-accent);
		text-decoration: underline;
		cursor: pointer;
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
