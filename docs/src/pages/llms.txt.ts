import type { APIRoute } from 'astro';
import { SITE, SUMMARY, sections, url } from '../llms';

export const GET: APIRoute = async () => {
	const { sections: all } = await sections();
	const lines = [`# BEAM`, '', `> ${SUMMARY}`, ''];
	for (const s of all) {
		lines.push(`## ${s.label}`, '');
		for (const p of s.pages) lines.push(`- [${p.data.title}](${url(p)}): ${p.data.description ?? ''}`);
		lines.push('');
	}
	lines.push('## Optional', '', `- [Every page in full](${SITE}llms-full.txt): the whole documentation as plain text, with each example's code and recorded output`, '');
	return new Response(lines.join('\n'), { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};
