import type { APIRoute } from 'astro';
import { SUMMARY, plain, sections, url } from '../llms';

export const GET: APIRoute = async () => {
	const { home, sections: all } = await sections();
	const parts = [`# BEAM`, `> ${SUMMARY}`];
	for (const p of [...(home ? [home] : []), ...all.flatMap((s) => s.pages)]) {
		parts.push(`---\n\n# ${p.data.title}\n\nURL: ${url(p)}\n\n> ${p.data.description ?? ''}\n\n${plain(p)}`);
	}
	return new Response(parts.join('\n\n') + '\n', { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};
