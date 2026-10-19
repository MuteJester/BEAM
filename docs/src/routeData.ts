import { defineRouteMiddleware } from '@astrojs/starlight/route-data';

// "BEAM" alone is also the Erlang virtual machine and Apache Beam, so every
// page's <title>, which is what search results show, says what this BEAM is.
// The site title in the header stays "BEAM".
const SUFFIX = 'BEAM: probability model for immune repertoires';

export const onRequest = defineRouteMiddleware((context) => {
	for (const entry of context.locals.starlightRoute.head) {
		if (entry.tag === 'title' && typeof entry.content === 'string') {
			entry.content = entry.content.replace(/ \| BEAM$/, ` | ${SUFFIX}`);
		}
	}
});
