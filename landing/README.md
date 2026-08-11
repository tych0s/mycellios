# mycellios landing page

Public website backed by the Control API and independent from the native node service, ready to be published at
`https://www.mycellios.com`.

## Development

```powershell
npm run landing:dev
```

## Validation and production build

```powershell
npm run landing:typecheck
npm run landing:build
npm run landing:preview
```

Static output is generated in `landing-dist/`. This is the directory the hosting
provider should make available to the coordinator. Public routes include
`robots.txt`, `sitemap.xml`, the brand favicon, and the Docusaurus manual under
`/docs/`.

The public documentation has an isolated source tree so internal research is
never published by accident:

```powershell
npm run docs:dev
npm run docs:build
```

`npm run landing:build` also runs the documentation build and writes its output
to `landing-dist/docs/`.

The coordinator owns the server-rendered `/blog` and `/blog/:slug` routes. Set
`CONTENT_HUB_API_URL` to the Content Hub API and configure the same
`MYCELLIOS_PUBLICATION_WEBHOOK_SECRET` in Mycellios and in the Content Hub
publication destination. Publication events are received at
`/api/content-hub/webhook`, invalidate the in-memory cache, and make new or
updated articles visible without rebuilding the landing page.

For a production-like local test, build the project and run the coordinator
with `MYCELLIOS_LANDING_DIST=./landing-dist`. Running only `landing:dev` serves
the static React landing and does not own the server-rendered blog routes.
Because the Mycellios coordinator uses port `8787`, start a local Content Hub
API with `PORT=8788` to match `.env.example`.

The interface and all SEO metadata are English-only so every visitor sees the
same public message.

The install section detects the visitor's operating system and downloads the
normalized assets from the latest public GitHub Release. Release filenames are
defined in `.github/workflows/node-build.yml`; keep those stable because the
landing uses `releases/latest/download` URLs.

## Visual assets

The two AI-generated editorial scenes are stored in:

- `src/assets/mycelium-network.jpg`
- `src/assets/efficiency-curve-ai.webp`

The hero is fully generative SVG and CSS. A branching mycelium colony grows once
across the viewport, reveals its crosslinks and fruiting bodies, and then rests.
It uses no animated raster, SVG turbulence, or perpetual background particles.
The editorial images are combined with lightweight SVG and CSS overlays.

Before publishing, confirm that `hello@mycellios.com` can receive email or update
the `EMAIL` constant in `src/rebrand/RebrandLanding.tsx` with the final address.
