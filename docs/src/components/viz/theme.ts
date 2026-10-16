/**
 * Shared helpers for the interactive figures: the site's colours (read from
 * Starlight's CSS variables, so charts follow light and dark mode), a theme
 * change subscription, data URLs under the site's base path, and number
 * formatting.
 */

export interface ThemeColors {
	accent: string;
	accentHigh: string;
	text: string;
	muted: string;
	grid: string;
	second: string;
}

export function themeColors(): ThemeColors {
	const style = getComputedStyle(document.documentElement);
	const v = (name: string) => style.getPropertyValue(name).trim();
	return {
		accent: v('--sl-color-accent'),
		accentHigh: v('--sl-color-accent-high'),
		text: v('--sl-color-white'),
		muted: v('--sl-color-gray-3'),
		grid: v('--sl-color-gray-5'),
		second: '#e07a3f',
	};
}

/** Calls `onChange` whenever the reader switches between light and dark; returns an unsubscribe. */
export function onThemeChange(onChange: () => void): () => void {
	const observer = new MutationObserver(onChange);
	observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
	return () => observer.disconnect();
}

/** URL of a data file written by an example under docs/public/data/. */
export function dataUrl(path: string): string {
	return `${import.meta.env.BASE_URL.replace(/\/$/, '')}/data/${path}`;
}

export function fmt(x: number, digits = 0): string {
	return x.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function pct(x: number, digits = 1): string {
	return `${(100 * x).toFixed(digits)}%`;
}

/** A probability as "3.3 × 10⁻⁵". */
export function sci(x: number, digits = 2): string {
	if (x === 0) return '0';
	const exp = Math.floor(Math.log10(x));
	const mant = x / 10 ** exp;
	const sup = String(exp)
		.split('')
		.map((c) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)])
		.join('');
	return `${mant.toFixed(digits - 1)} × 10${sup}`;
}
