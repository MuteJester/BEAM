<script lang="ts">
	/**
	 * A sequence drawn as tiles, with an expert's window over it: the k residues of context the
	 * expert has just read (tinted) and the residue it predicts (framed). The forward expert's
	 * window reaches back towards the start, the backward expert's forward towards the end; past
	 * either end the context is padding, drawn as ghost tiles. The window slides as its target
	 * moves. Above the tiles each residue's position from the start, below from the end.
	 */
	interface Window {
		target: number; // index of the residue being predicted
		dir: 'f' | 'b';
	}
	interface Props {
		sequence: string;
		k: number;
		windows: Window[];
		hidden?: boolean; // the predicted residue is shown as "?"
		// optional bars under each residue, values in [0, 1]: the forward expert's rise from a
		// centre line and the backward expert's hang below it
		strips?: { dir: 'f' | 'b'; values: (number | null)[]; label?: string }[];
		onpick?: (i: number) => void;
		minLength?: number; // lay out at least this many residues, so tiles keep their size across sequences
	}
	let { sequence, k, windows, hidden = false, strips, onpick, minLength = 0 }: Props = $props();

	let width = $state(640);
	const L = $derived(sequence.length);
	const n = $derived(L + 2 * k); // k ghost slots on each side keep the layout fixed
	const gap = $derived(width < 480 ? 2 : 4);
	const slots = $derived(Math.max(L, minLength) + 2 * k);
	const tile = $derived(Math.max(13, Math.min(34, (width - (slots - 1) * gap) / slots)));
	const pitch = $derived(tile + gap);
	const x = (slot: number) => slot * pitch;

	const targets = $derived(new Set(windows.map((w) => w.target)));
	const ghostLeft = $derived(windows.some((w) => w.dir === 'f' && w.target - k < 0));
	const ghostRight = $derived(windows.some((w) => w.dir === 'b' && w.target + k > L - 1));
	const active = $derived(new Set(windows.map((w) => w.dir)));
	// Two windows on one residue: each frames only its context, and the white tile marks the residue.
	const shared = $derived(targets.size < windows.length);

	function frame(w: Window) {
		const first = w.dir === 'f' ? w.target - k : w.target;
		return { left: x(first + k) - 3, width: (k + 1) * pitch - gap + 6 };
	}
</script>

<div class="lens" bind:clientWidth={width} style:--tile={`${tile}px`} style:--gap={`${gap}px`}>
	<div class="track" style:width={`${n * pitch - gap}px`}>
		<div class="counter top" class:on={active.has('f')} aria-hidden="true">
			{#if width >= 480}<span class="rowlab" style:width={`${k * pitch - gap}px`}>from the start</span>{/if}
			{#each [...sequence] as _, i}
				<span class="num" class:cur={windows.some((w) => w.dir === 'f' && w.target === i)} style:left={`${x(i + k)}px`}>{i + 1}</span>
			{/each}
		</div>

		<div class="tiles">
			{#each Array(k) as _, j}
				<span class="tile ghost" class:show={ghostLeft} style:left={`${x(j)}px`}>·</span>
			{/each}
			{#each [...sequence] as ch, i}
				{#if onpick}
					<button
						class="tile res"
						class:target={targets.has(i)}
						style:left={`${x(i + k)}px`}
						aria-label={`Residue ${i + 1}, ${ch}`}
						aria-pressed={targets.has(i)}
						onclick={() => onpick(i)}>{hidden && targets.has(i) ? '?' : ch}</button
					>
				{:else}
					<span class="tile res" class:target={targets.has(i)} style:left={`${x(i + k)}px`}>{hidden && targets.has(i) ? '?' : ch}</span>
				{/if}
			{/each}
			{#each Array(k) as _, j}
				<span class="tile ghost" class:show={ghostRight} style:left={`${x(L + k + j)}px`}>·</span>
			{/each}
			{#each windows as w (w.dir)}
				{@const f = frame(w)}
				<span class="frame {w.dir}" class:shared style:left={`${f.left}px`} style:width={`${f.width}px`} aria-hidden="true">
					{#if w.dir === 'f'}
						<span class="ctx" style:width={`${k * pitch}px`}></span><span class="tgt"></span>
					{:else}
						<span class="tgt"></span><span class="ctx" style:width={`${k * pitch}px`}></span>
					{/if}
				</span>
			{/each}
		</div>

		<div class="counter bottom" class:on={active.has('b')} aria-hidden="true">
			{#if width >= 480}<span class="rowlab" style:width={`${k * pitch - gap}px`}>from the end</span>{/if}
			{#each [...sequence] as _, i}
				<span class="num" class:cur={windows.some((w) => w.dir === 'b' && w.target === i)} style:left={`${x(i + k)}px`}>{L - i}</span>
			{/each}
		</div>

		{#if strips}
			<div class="strips" aria-hidden="true">
				{#each strips as st (st.dir)}
					<div class="half {st.dir}">
						{#if width >= 480 && st.label}<span class="rowlab" style:width={`${k * pitch - gap}px`}>{st.label}</span>{/if}
						{#each st.values as v, i}
							{#if v !== null}
								<span class="sbar" style:left={`${x(i + k) + tile * 0.2}px`} style:height={`${Math.max(2, 100 * v)}%`}></span>
							{/if}
						{/each}
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>

<style>
	.lens {
		--f: var(--sl-color-accent);
		--b: #ef785f;
		width: 100%;
		overflow: hidden;
	}
	.track {
		position: relative;
		margin-inline: auto;
	}
	.tiles,
	.counter,
	.half {
		position: relative;
	}
	.tiles {
		height: calc(var(--tile) + 10px);
	}
	.tile {
		position: absolute;
		top: 5px;
		width: var(--tile);
		height: var(--tile);
		display: grid;
		place-items: center;
		border-radius: 0.3rem;
		font-family: var(--sl-font-mono);
		font-size: calc(var(--tile) * 0.5);
		font-weight: 600;
		transition: left 0.3s ease;
	}
	.res {
		background: var(--sl-color-gray-6);
		color: var(--sl-color-white);
		border: 0;
		padding: 0;
	}
	button.res {
		cursor: pointer;
	}
	button.res:hover {
		outline: 1px solid var(--sl-color-gray-3);
	}
	.res.target {
		background: var(--sl-color-white);
		color: var(--sl-color-black);
	}
	.ghost {
		color: var(--sl-color-gray-3);
		border: 1px dashed var(--sl-color-gray-5);
		opacity: 0;
		transition: opacity 0.3s ease;
	}
	.ghost.show {
		opacity: 1;
	}
	.frame {
		position: absolute;
		top: 2px;
		height: calc(var(--tile) + 6px);
		display: flex;
		pointer-events: none;
		transition: left 0.3s ease;
	}
	.ctx {
		height: 100%;
		border: 2px solid var(--c);
		background: color-mix(in srgb, var(--c) 22%, transparent);
	}
	.tgt {
		flex: 1;
		height: 100%;
		border: 3px solid var(--c);
	}
	.frame.f {
		--c: var(--f);
	}
	.frame.b {
		--c: var(--b);
	}
	.frame.f .ctx {
		border-radius: 0.4rem 0 0 0.4rem;
		border-right: 0;
	}
	.frame.f .tgt {
		border-radius: 0 0.4rem 0.4rem 0;
	}
	.frame.b .ctx {
		border-radius: 0 0.4rem 0.4rem 0;
		border-left: 0;
	}
	.frame.b .tgt {
		border-radius: 0.4rem 0 0 0.4rem;
	}
	.frame.shared .tgt {
		border-color: transparent;
	}
	.frame.shared .ctx {
		border: 2px solid var(--c);
		border-radius: 0.4rem;
	}
	.counter {
		height: 1rem;
		font-size: 10px;
		color: var(--sl-color-gray-4);
	}
	.counter.on {
		color: var(--sl-color-gray-2);
	}
	.num {
		position: absolute;
		width: var(--tile);
		text-align: center;
		transition: color 0.2s ease;
	}
	.counter.top .num.cur {
		color: var(--f);
		font-weight: 700;
	}
	.counter.bottom .num.cur {
		color: var(--b);
		font-weight: 700;
	}
	.rowlab {
		position: absolute;
		left: 0;
		font-size: 10px;
		color: var(--sl-color-gray-3);
		text-align: right;
		white-space: nowrap;
		font-style: italic;
	}
	.strips {
		margin-top: 0.2rem;
	}
	.half {
		height: 1.9rem;
	}
	.half.f {
		border-bottom: 1px solid var(--sl-color-gray-4);
	}
	.half.f .rowlab {
		bottom: 0.15rem;
	}
	.half.b .rowlab {
		top: 0.15rem;
	}
	.sbar {
		position: absolute;
		width: calc(var(--tile) * 0.6);
		transition: height 0.3s ease;
	}
	.half.f .sbar {
		bottom: 0;
		border-radius: 2px 2px 0 0;
		background: var(--f);
	}
	.half.b .sbar {
		top: 0;
		border-radius: 0 0 2px 2px;
		background: var(--b);
	}
</style>
