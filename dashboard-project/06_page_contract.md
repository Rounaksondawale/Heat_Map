# Page contract for the published dashboard (from the Artifact host)

The page is published as a claude.ai Artifact. These rules are enforced by the host.

- Author ONE `.html` file with no `<!doctype>`, `<html>`, `<head>`, `<body>` tags: the host wraps it. Start the file with `<title>Heat_Map Results</title>`, `<meta name="claude:viewport-fit-sides" content="cover">`, then `<style>`.
- Title is a name (2-4 words). Keep `Heat_Map Results`.
- External scripts ONLY from cdnjs.cloudflare.com (preferred), cdn.jsdelivr.net/npm/, unpkg.com; pin exact versions (e.g. `.../d3/7.9.0/d3.min.js`). Stylesheets ONLY from fonts.googleapis.com (with fallback stacks). Everything else (fetch, images from other hosts, iframes) is blocked silently. Inline all data as JS constants; embed images as data: URIs only if small (the docs PNGs may be embedded if total page < 3 MB; otherwise link to GitHub).
- Mermaid renders natively via `<pre class="mermaid">`, but prefer a hand-drawn inline SVG pipeline diagram for theme control.
- No `alert/confirm/prompt`, no `window.print()`, no download links, no iframes.
- `localStorage` allowed only for the theme preference, wrapped in try/catch.
- Hash routing: only plain `#name` anchors (ASCII, hyphens). Never use `settings`, `sidebar`, `customize`, `upgrade`, `downloads`, `comment-*`.
- Responsive: works at 400px wide, no horizontal body scroll. Side gutter on the outer wrapper: `padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));` with vertical spacing via `padding-block`. Tables and wide charts go inside their own `overflow-x: auto` containers. Flex/grid children holding text get `min-width: 0`.
- Sticky nav: `position: sticky; top: env(safe-area-inset-top, 0px);` and add `body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--nav-bg); }`.
- Theme: every colour is a token. Shape:
  ```css
  :root { /* all light tokens */ }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { /* dark tokens */ color-scheme: dark } }
  :root[data-theme="dark"] { /* same dark tokens */ color-scheme: dark }
  body { background: var(--bg); color: var(--fg) }
  ```
  The toggle sets `document.documentElement.dataset.theme = 'light'|'dark'`. Charts must read colours from CSS variables (getComputedStyle) and re-render on theme change.
- Page must be complete at rest: no content hidden behind scroll-triggered animations. Respect `prefers-reduced-motion`. Visible keyboard focus. `font-variant-numeric: tabular-nums` on numbers.
- Design: avoid templated looks (cream+serif+terracotta, purple gradient hero, emoji section markers, everything centred, same rounded card everywhere). Hero sized to content, not 100vh. Copy is plain, active voice, no em-dash asides, no "worth noting".
