// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import svelte from '@astrojs/svelte';
import starlightLinksValidator from 'starlight-links-validator';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';

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
				'Exact probabilities for T cell receptor repertoires: sequence probabilities, sequencing depth, publicness and repertoire comparison.',
			logo: {
				light: './src/assets/logo-light.svg',
				dark: './src/assets/logo-dark.svg',
			},
			favicon: '/favicon.svg',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/MuteJester/BEAM' }],
			editLink: { baseUrl: 'https://github.com/MuteJester/BEAM/edit/main/docs/' },
			customCss: ['katex/dist/katex.min.css', './src/styles/theme.css'],
			lastUpdated: true,
			plugins: [starlightLinksValidator()],
			sidebar: [
				{ label: 'Get started', items: [{ autogenerate: { directory: 'start' } }] },
				{ label: 'Guides', items: [{ autogenerate: { directory: 'guides' } }] },
				{ label: 'Explanations', items: [{ autogenerate: { directory: 'explanations' } }] },
				{ label: 'Reference', collapsed: true, items: [{ autogenerate: { directory: 'reference' } }] },
				{ label: 'Resources', collapsed: true, items: [{ autogenerate: { directory: 'resources' } }] },
			],
		}),
		svelte(),
	],
});
