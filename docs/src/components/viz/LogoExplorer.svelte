<script lang="ts">
	import { dataUrl, pct } from './theme';

	interface Logo {
		bits: number[][];
		consensus: string;
		fraction: number;
		information: number;
	}
	interface Data {
		residues: string;
		all: Record<string, Logo>;
		common: Record<string, Logo>;
		rare: Record<string, Logo>;
	}

	let data = $state<Data | null>(null);
	let length = $state(15);
	let view = $state<'all' | 'common' | 'rare'>('all');
	let failed = $state(false);

	$effect(() => {
		fetch(dataUrl('viz/logos/logos.json'))
			.then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
			.then((d: Data) => (data = d))
			.catch(() => (failed = true));
	});

	const hasSplit = $derived(!!data?.common[length]);
	// The view actually shown: common and rare exist only for some lengths.
	const shown = $derived(hasSplit ? view : 'all');
	const logo = $derived(data ? data[shown][length] : null);
	const COLOURS: Record<string, string> = {
		...Object.fromEntries([...'AVLIM'].map((a) => [a, '#8a9a2b'])),
		...Object.fromEntries([...'FWY'].map((a) => [a, '#6c7bd9'])),
		...Object.fromEntries([...'STNQC'].map((a) => [a, '#14a394'])),
		...Object.fromEntries([...'KRH'].map((a) => [a, '#c4568a'])),
		...Object.fromEntries([...'DE'].map((a) => [a, '#ef785f'])),
		G: '#7d7d7d',
		P: '#b08a2e',
	};
	const H = 160; // pixels for the tallest stack
	const maxBits = $derived(logo ? Math.max(2, ...logo.bits.map((c) => c.reduce((a, b) => a + b, 0))) : 2);
	// Letters stacked from smallest to largest; each scaled vertically to its height.
	const stacks = $derived(
		logo && data
			? logo.bits.map((col) => {
					let y = 0;
					return col
						.map((h, a) => ({ h, a }))
						.filter((d) => d.h > 0.02)
						.sort((p, q) => p.h - q.h)
						.map(({ h, a }) => {
							const out = { letter: data!.residues[a], h: (h / maxBits) * H, y };
							y += out.h;
							return out;
						});
				})
			: [],
	);
</script>

<div class="logos">
	{#if data && logo}
		<label class="control">
			<span>Length: <strong>{length}</strong> residues</span>
			<input type="range" min="10" max="20" step="1" bind:value={length} aria-label="CDR3 length" />
		</label>
		<div class="row" role="group" aria-label="Which sequences">
			<button class:active={shown === 'all'} aria-pressed={shown === 'all'} onclick={() => (view = 'all')}>all sequences</button>
			<button class:active={shown === 'common'} aria-pressed={shown === 'common'} disabled={!hasSplit} onclick={() => (view = 'common')}>common (P &gt; 10⁻⁷)</button>
			<button class:active={shown === 'rare'} aria-pressed={shown === 'rare'} disabled={!hasSplit} onclick={() => (view = 'rare')}>rare (P ≤ 10⁻⁷)</button>
			{#if !hasSplit}<span class="muted">(common and rare are shown for lengths 13 to 16)</span>{/if}
		</div>
		<svg viewBox={`0 0 ${logo.bits.length * 30 + 10} ${H + 24}`} role="img" aria-label={`Sequence logo, length ${length}`}>
			{#each stacks as stack, i}
				{#each stack as l}
					<text
						x="0"
						y="0"
						font-family="DejaVu Sans, Arial, sans-serif"
						font-weight="700"
						font-size="34"
						fill={COLOURS[l.letter] ?? '#7d7d7d'}
						text-anchor="middle"
						transform={`translate(${i * 30 + 20}, ${H - l.y}) scale(1, ${l.h / 25})`}
					>{l.letter}</text>
				{/each}
				<text x={i * 30 + 20} y={H + 18} font-size="10" text-anchor="middle" fill="currentColor">{i + 1}</text>
			{/each}
		</svg>
		<p class="readout">
			Consensus <span class="mono">{logo.consensus}</span>; information {logo.information.toFixed(1)} bits{shown !== 'all'
				? `; these sequences carry ${pct(logo.fraction)} of the probability at length ${length}`
				: ''}.
		</p>
	{:else if failed}
		<p class="muted">The data for this figure could not be loaded.</p>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.logos {
		display: grid;
		gap: 0.6rem;
	}
	.control {
		display: grid;
		gap: 0.35rem;
	}
	input[type='range'] {
		width: 100%;
		accent-color: var(--sl-color-accent);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: center;
	}
	button {
		flex: none;
		font: inherit;
		font-size: var(--sl-text-xs);
		padding: 0.2rem 0.6rem;
		border-radius: 999px;
		border: 1px solid var(--sl-color-gray-5);
		background: transparent;
		color: var(--sl-color-white);
		cursor: pointer;
	}
	button:disabled {
		opacity: 0.4;
		cursor: default;
	}
	button.active {
		background: var(--sl-color-accent);
		border-color: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	svg {
		width: 100%;
		height: auto;
		color: var(--sl-color-gray-3);
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
		font-size: var(--sl-text-sm);
	}
</style>
