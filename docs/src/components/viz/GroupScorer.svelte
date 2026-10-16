<script lang="ts">
	import * as Plot from '@observablehq/plot';
	import { dataUrl, onThemeChange, pct, themeColors } from './theme';

	interface Row {
		sequence: string;
		drawn_from: string;
		length: number;
		motif: boolean;
		log_p: Record<string, number>;
	}

	let data = $state<{ motif: string; rows: Row[]; accuracy: Record<string, number> } | null>(null);
	let chosen = $state(0);
	let themeTick = $state(0);
	let failed = $state(false);
	let chart: HTMLDivElement | undefined = $state();
	const groups = ['A', 'B', 'C'];

	$effect(() => {
		fetch(dataUrl('viz/group_scorer/group_scorer.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d) => (data = d))
			.catch(() => (failed = true));
		return onThemeChange(() => themeTick++);
	});

	const row = $derived(data?.rows[chosen]);
	// Equal prior on the three groups: the posterior is a softmax of the log probabilities.
	const posterior = $derived.by(() => {
		if (!row) return {} as Record<string, number>;
		const m = Math.max(...groups.map((g) => row.log_p[g]));
		const e = groups.map((g) => Math.exp(row.log_p[g] - m));
		const z = e.reduce((a, b) => a + b, 0);
		return Object.fromEntries(groups.map((g, i) => [g, e[i] / z]));
	});
	const best = $derived(row ? groups.reduce((a, b) => (row.log_p[b] > row.log_p[a] ? b : a)) : 'A');

	$effect(() => {
		themeTick;
		if (!row || !chart) return;
		const c = themeColors();
		const colour = { A: c.accent, B: c.second, C: '#6c7bd9' } as Record<string, string>;
		const plot = Plot.plot({
			width: chart.clientWidth,
			height: 170,
			marginLeft: 70,
			style: { color: c.text, background: 'transparent', fontSize: '12px' },
			x: { label: '← log probability relative to the most probable group (nats)', grid: true },
			y: { domain: groups, label: null, tickFormat: (g) => `group ${g}` },
			marks: [
				Plot.barX(groups, {
					y: (g) => g,
					x1: 0,
					x2: (g) => row.log_p[g] - Math.max(...groups.map((h) => row.log_p[h])),
					fill: (g) => colour[g],
				}),
				Plot.ruleX([0], { stroke: c.muted }),
			],
		});
		chart.replaceChildren(plot);
	});
</script>

<div class="scorer">
	{#if data && row}
		{#each groups as g}
			<div class="row" role="group" aria-label={`Sequences drawn from group ${g}`}>
				<span class="muted">from group {g}</span>
				{#each data.rows as r, k}
					{#if r.drawn_from === g}
						<button class="mono" class:active={chosen === k} aria-pressed={chosen === k} onclick={() => (chosen = k)}>{r.sequence}</button>
					{/if}
				{/each}
			</div>
		{/each}
		<div bind:this={chart} role="img" aria-label="Log probability of the chosen sequence under each group's model"></div>
		<p class="readout" aria-live="polite">
			<span class="mono">{row.sequence}</span> ({row.length} residues{row.motif ? `, carries ${data.motif}` : ''}):
			most probable under group <strong>{best}</strong>; it was drawn from group {row.drawn_from}. If the three group
			models were exact and the groups equally likely beforehand, the chance it came from A, B or C would be
			{pct(posterior.A)}, {pct(posterior.B)} and {pct(posterior.C)}.
		</p>
		<p class="readout muted">
			Over all unseen sequences of a new donor, the group under which a sequence is most probable is its own group for
			{pct(data.accuracy.A)} of A's, {pct(data.accuracy.B)} of B's and {pct(data.accuracy.C)} of C's sequences: one sequence
			says little about its group.
		</p>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.scorer {
		display: grid;
		gap: 0.5rem;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		align-items: center;
	}
	.row .muted {
		min-width: 6.5rem;
		font-size: var(--sl-text-sm);
	}
	button {
		flex: none;
		font: inherit;
		font-size: var(--sl-text-xs);
		padding: 0.15rem 0.55rem;
		border-radius: 999px;
		border: 1px solid var(--sl-color-gray-5);
		background: transparent;
		color: var(--sl-color-white);
		cursor: pointer;
	}
	button.active {
		background: var(--sl-color-accent);
		border-color: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.readout {
		font-size: var(--sl-text-sm);
		margin: 0;
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
