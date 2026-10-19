// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import svelte from '@astrojs/svelte';
import starlightLinksValidator from 'starlight-links-validator';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';
import fs from 'node:fs';

/** Whether a docs section has a page that is not hidden from the sidebar. */
function hasPages(dir) {
	const path = new URL(`./src/content/docs/${dir}/`, import.meta.url);
	if (!fs.existsSync(path)) return false;
	return fs
		.readdirSync(path)
		.filter((f) => /\.mdx?$/.test(f))
		.some((f) => !/^\s*hidden:\s*true/m.test(fs.readFileSync(new URL(f, path), 'utf8')));
}

const SITE_URL = 'https://mutejester.github.io/BEAM/';

// schema.org description of the package, for search engines (one JSON-LD block on every page).
const STRUCTURED_DATA = {
	'@context': 'https://schema.org',
	'@type': 'SoftwareSourceCode',
	name: 'BEAM',
	alternateName: 'Bidirectional Expert-Anchored Model',
	description:
		'A probability model for biological sequences such as T cell receptor CDR3s, learned from sequences alone with optional abundances: sequence probabilities, sequencing depth, publicness and repertoire comparison, summed over every possible sequence without simulation.',
	url: SITE_URL,
	codeRepository: 'https://github.com/MuteJester/BEAM',
	identifier: 'https://doi.org/10.5281/zenodo.23470065',
	programmingLanguage: ['Python', 'C'],
	runtimePlatform: 'Python 3.11+',
	license: 'https://spdx.org/licenses/GPL-3.0-or-later.html',
	keywords: [
		'T cell receptor', 'TCR repertoire', 'CDR3', 'AIRR-seq', 'immune repertoire',
		'probabilistic model', 'sequence probability', 'publicness', 'sampling depth',
		'repertoire comparison', 'bioinformatics',
	],
	author: [
		{ '@type': 'Person', givenName: 'Thomas', familyName: 'Konstantinovsky' },
		{ '@type': 'Person', givenName: 'Gur', familyName: 'Yaari' },
	],
};

// Served from GitHub Pages at https://mutejester.github.io/BEAM/, so every
// internal URL carries the /BEAM base; components build asset and data URLs
// from import.meta.env.BASE_URL rather than writing absolute paths.
export default defineConfig({
	site: 'https://mutejester.github.io',
	base: '/BEAM',
	markdown: {
		// Equations ($...$ and $$...$$) are typeset by KaTeX at build time.
		processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }),
	},
	integrations: [
		starlight({
			title: 'BEAM',
			description:
				'A probability model for biological sequences such as T cell receptor CDR3s: sequence probabilities, sequencing depth, publicness and repertoire comparison, summed over every possible sequence without simulation.',
			logo: {
				light: './src/assets/logo-light.svg',
				dark: './src/assets/logo-dark.svg',
			},
			favicon: '/favicon.svg',
			head: [
				{ tag: 'link', attrs: { rel: 'icon', href: '/BEAM/favicon.ico', sizes: '32x32' } },
				// The card shown when a page is shared (Slack, X, LinkedIn, ...).
				{ tag: 'meta', attrs: { property: 'og:image', content: `${SITE_URL}social-card.png` } },
				{ tag: 'meta', attrs: { property: 'og:image:width', content: '1200' } },
				{ tag: 'meta', attrs: { property: 'og:image:height', content: '630' } },
				{ tag: 'meta', attrs: { property: 'og:image:alt', content: 'BEAM: every possible sequence, accounted for' } },
				{ tag: 'meta', attrs: { name: 'twitter:image', content: `${SITE_URL}social-card.png` } },
				// Tells search engines that this is a software package and where its code is.
				{ tag: 'script', attrs: { type: 'application/ld+json' }, content: JSON.stringify(STRUCTURED_DATA) },
			],
			// Every page's <title> says what this BEAM is (src/routeData.ts).
			routeMiddleware: './src/routeData.ts',
			// Section links beside the search box, before the GitHub icon.
			// "On this page" drawn as a progress rail (src/components/Toc.astro).
			components: {
				SocialIcons: './src/components/HeaderLinks.astro',
				TableOfContents: './src/components/Toc.astro',
			},
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/MuteJester/BEAM' }],
			editLink: { baseUrl: 'https://github.com/MuteJester/BEAM/edit/main/docs/' },
			customCss: ['katex/dist/katex.min.css', './src/styles/theme.css'],
			lastUpdated: true,
			plugins: [starlightLinksValidator()],
			// A section appears once it has a visible page, so the sidebar never
			// shows an empty heading while the site grows.
			sidebar: [
				['Get started', 'start'],
				['Guides', 'guides'],
				['Explanations', 'explanations'],
				['Reference', 'reference'],
				['Resources', 'resources'],
			]
				.filter(([, dir]) => hasPages(dir))
				.map(([label, dir]) => ({
					label,
					collapsed: dir === 'reference' || dir === 'resources',
					items: [{ autogenerate: { directory: dir } }],
				})),
		}),
		svelte(),
	],
});
