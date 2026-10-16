<script lang="ts">
	import { dataUrl, sci } from './theme';

	interface Seq {
		sequence: string;
		start_bits: number[];
		end_bits: number[];
		log_length: number;
		log_partition: number;
		log_p: number;
	}
	interface Data {
		weights: [number, number];
		sequences: Seq[];
	}

	let data = $state<Data | null>(null);
	let chosen = $state(0);
	let step = $state(0); // residues revealed so far
	let playing = $state(false);

	$effect(() => {
		fetch(dataUrl('viz/two_anchors/two_anchors.json'))
			.then((r) => r.json())
			.then((d: Data) => (data = d));
	});

	const seq = $derived(data?.sequences[chosen]);
	const length = $derived(seq ? seq.sequence.length : 0);
	const maxBits = $derived(seq ? Math.max(...seq.start_bits, ...seq.end_bits, 1) : 1);
	// Mixed surprisal of the revealed residues, in bits (weights from the model).
	const residueBits = $derived(
		seq && data
			? seq.start_bits
					.slice(0, step)
					.reduce((acc, b, i) => acc + data!.weights[0] * b + data!.weights[1] * seq.end_bits[i], 0)
			: 0,
	);
	const done = $derived(step >= length);

	$effect(() => {
		if (!playing) return;
		const timer = setInterval(() => {
			if (step >= length) {
				playing = false;
			} else {
				step += 1;
			}
		}, 650);
		return () => clearInterval(timer);
	});

	function choose(i: number) {
		chosen = i;
		step = 0;
		playing = true;
	}
	const LN2 = Math.LN2;
</script>

<div class="anchors">
	{#if data && seq}
		<div class="row" role="group" aria-label="Sequence">
			{#each data.sequences as s, i}
				<button class="mono" class:active={chosen === i} onclick={() => choose(i)}>{s.sequence}</button>
			{/each}
		</div>
		<div class="controls">
			<button
				onclick={() => {
					if (done) step = 0;
					playing = !playing || done;
				}}>{playing ? 'Pause' : done ? 'Replay' : step ? 'Resume' : 'Play'}</button
			>
			<button
				onclick={() => {
					playing = false;
					step = Math.max(0, step - 1);
				}}>◀ Step</button
			>
			<button
				onclick={() => {
					playing = false;
					step = Math.min(length, step + 1);
				}}>Step ▶</button
			>
		</div>

		<div class="grid" style={`--n: ${length}`}>
			{#each seq.sequence.split('') as res, i}
				<div class="col" class:current={i === step - 1} class:hidden={i >= step}>
					<div class="bars">
						<div
							class="bar start"
							style={`height: ${(100 * seq.start_bits[i]) / maxBits}%`}
							title={`read from the start: ${seq.start_bits[i].toFixed(2)} bits`}
						></div>
						<div
							class="bar end"
							style={`height: ${(100 * seq.end_bits[i]) / maxBits}%`}
							title={`read from the end: ${seq.end_bits[i].toFixed(2)} bits`}
						></div>
					</div>
					<div class="res mono">{res}</div>
					<div class="idx">{i}·{length - 1 - i}</div>
				</div>
			{/each}
		</div>
		<div class="legend">
			<span><span class="swatch start"></span>read from the start (position counted from the front)</span>
			<span><span class="swatch end"></span>read from the end (position counted from the back)</span>
		</div>

		<div class="readout" aria-live="polite">
			{#if step > 0 && !done}
				Residue {step} of {length}, <span class="mono">{seq.sequence[step - 1]}</span>:
				{seq.start_bits[step - 1].toFixed(2)} bits of surprise read from the start,
				{seq.end_bits[step - 1].toFixed(2)} bits read from the end.
			{:else if done}
				<div>
					ln P = ln π(L) ({seq.log_length.toFixed(2)}) + residues ({(-residueBits * LN2).toFixed(2)}) − ln Z(L)
					({seq.log_partition.toFixed(2)}) = <strong>{seq.log_p.toFixed(2)}</strong>
				</div>
				<div>P(<span class="mono">{seq.sequence}</span>) = <strong>{sci(Math.exp(seq.log_p))}</strong></div>
			{:else}
				Press Play to read the sequence one residue at a time.
			{/if}
		</div>
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</div>

<style>
	.anchors {
		display: grid;
		gap: 0.75rem;
	}
	.row,
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
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
	button.active {
		background: var(--sl-color-accent);
		border-color: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	.mono {
		font-family: var(--sl-font-mono);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(var(--n), minmax(0, 1fr));
		gap: 0.15rem;
		align-items: end;
	}
	.col {
		display: grid;
		justify-items: center;
		transition: opacity 0.3s;
	}
	.col.hidden .bars {
		opacity: 0.12;
	}
	.col.current .res {
		background: var(--sl-color-accent);
		color: var(--sl-color-black);
	}
	.bars {
		display: flex;
		gap: 2px;
		align-items: flex-end;
		height: 9rem;
		width: 100%;
		justify-content: center;
		transition: opacity 0.3s;
	}
	.bar {
		width: 40%;
		min-height: 2px;
		border-radius: 2px 2px 0 0;
		transition: height 0.4s;
	}
	.bar.start,
	.swatch.start {
		background: var(--sl-color-accent);
	}
	.bar.end,
	.swatch.end {
		background: #ef785f;
	}
	.res {
		font-size: var(--sl-text-base);
		font-weight: 600;
		padding: 0 0.25rem;
		border-radius: 0.25rem;
	}
	.idx {
		font-size: 0.65rem;
		color: var(--sl-color-gray-3);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		font-size: var(--sl-text-xs);
		color: var(--sl-color-gray-3);
	}
	.swatch {
		display: inline-block;
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 2px;
		margin-right: 0.35rem;
	}
	.readout {
		font-size: var(--sl-text-sm);
		min-height: 2.8rem;
	}
	.muted {
		color: var(--sl-color-gray-3);
	}
</style>
