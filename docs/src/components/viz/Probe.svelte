<script lang="ts">
	// Style-test only: proves an island hydrates and fetches data under the base path.
	let total = $state<number | null>(null);
	$effect(() => {
		fetch(`${import.meta.env.BASE_URL.replace(/\/$/, '')}/data/styletest/demo/squares.json`)
			.then((r) => r.json())
			.then((d) => (total = d.square.reduce((a: number, b: number) => a + b, 0)));
	});
</script>

<p data-probe>Sum fetched in the browser: {total ?? 'loading'}</p>
