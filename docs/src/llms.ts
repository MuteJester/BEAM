/**
 * The documentation as plain text for language models and AI assistants
 * (https://llmstxt.org): /llms.txt is an index of every page, /llms-full.txt
 * every page in full, with each example's code and its recorded output in place
 * of the components that show them on the website.
 */
import fs from 'node:fs';
import path from 'node:path';
import { getCollection, type CollectionEntry } from 'astro:content';

export const SITE = 'https://mutejester.github.io/BEAM/';
export const SUMMARY =
	'BEAM is a probability model of biological sequences, amino acid or nucleotide, learned from the ' +
	'sequences alone, with their abundances if available. It gives the probability of any sequence, ' +
	'the expected number of distinct sequences at a sequencing depth and how many two samples share, ' +
	'how many people in a cohort carry a sequence, and distances between repertoires split into ' +
	'length and local-pattern parts, all summed over every possible sequence without simulation. A ' +
	'reference model of human TCRβ CDR3s built from 38,538 repertoires is published with it. Python ' +
	'package: beam-model (import BEAM).';

const SECTIONS: [string, string][] = [
	['start', 'Get started'],
	['guides', 'Guides'],
	['explanations', 'Explanations'],
	['reference', 'Reference'],
	['resources', 'Resources'],
];

type Page = CollectionEntry<'docs'>;

export const url = (page: Page) => (page.id === 'index' ? SITE : `${SITE}${page.id}/`);

/** The pages of each section, in sidebar order. */
export async function sections() {
	const pages = (await getCollection('docs')).filter((p) => !p.data.sidebar?.hidden && p.id !== '404');
	const order = (p: Page) => p.data.sidebar?.order ?? 999;
	return {
		home: pages.find((p) => p.id === 'index'),
		sections: SECTIONS.map(([dir, label]) => ({
			label,
			pages: pages
				.filter((p) => p.id.startsWith(`${dir}/`))
				.sort((a, b) => order(a) - order(b) || a.data.title.localeCompare(b.data.title)),
		})).filter((s) => s.pages.length),
	};
}

const GENERATED = path.join(process.cwd(), 'src/generated');
const read = (file: string) => (fs.existsSync(file) ? fs.readFileSync(file, 'utf8').trim() : '');
const attrs = (s: string) => Object.fromEntries([...s.matchAll(/(\w+)="([^"]*)"/g)].map((m) => [m[1], m[2]]));
const absolute = (href: string) => (href.startsWith('/BEAM/') ? SITE + href.slice('/BEAM/'.length) : href);

function example(src: string) {
	const code = read(path.join(GENERATED, src, 'code.py'));
	const output = read(path.join(GENERATED, src, 'output.txt'));
	return `\`\`\`python\n${code}\n\`\`\`\n\nOutput:\n\n\`\`\`\n${output}\n\`\`\``;
}

/** A page's source as plain Markdown: components replaced by what they show. */
export function plain(page: Page) {
	return (page.body ?? '')
		.replace(/^import\s+[\w{},\s*]+\s+from\s+['"][^'"]+['"];?\s*$/gm, '')
		.replace(/<Interactive\b([\s\S]*?)>[\s\S]*?<\/Interactive>/g, (_, a) => {
			const { title, description } = attrs(a);
			return `*Interactive figure on the website, "${title}":* ${description}`;
		})
		.replace(/<Example\s+src="([^"]+)"[^>]*\/>/g, (_, src) => example(src))
		.replace(/<Figure\b([\s\S]*?)\/>/g, (_, a) => {
			const { caption, alt } = attrs(a);
			return `*Figure: ${caption ?? alt}*`;
		})
		.replace(/<Aside\b([^>]*)>([\s\S]*?)<\/Aside>/g, (_, a, body) => {
			const { title, type } = attrs(a);
			return `**${title ?? type ?? 'Note'}:** ${body.trim()}`;
		})
		.replace(/<LinkCard\b([\s\S]*?)\/>/g, (_, a) => {
			const { title, description, href } = attrs(a);
			return `- [${title}](${absolute(href)}): ${description}`;
		})
		.replace(/^\s*<\/?[A-Z][\w.]*[^>]*>\s*$/gm, '')
		.replace(/\]\(\/BEAM\//g, `](${SITE}`)
		.replace(/\n{3,}/g, '\n\n')
		.trim();
}
