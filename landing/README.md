# mycellios landing page

Public website independent from the Electron client, ready to be published at
`https://mycellios.com`.

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
provider should publish. Public routes include `robots.txt`, `sitemap.xml`, and
the brand favicon.

The interface and all SEO metadata are English-only so every visitor sees the
same public message.

The install section detects the visitor's operating system and downloads the
normalized assets from the latest public GitHub Release. Release filenames are
defined in `.github/workflows/desktop-build.yml`; keep those stable because the
landing uses `releases/latest/download` URLs.

## Visual assets

The two AI-generated editorial scenes are stored in:

- `src/assets/mycelium-network.jpg`
- `src/assets/efficiency-curve-ai.webp`

The hero is fully generative SVG and CSS: organic cell membranes, filaments,
light fields, and moving information packets are rendered without a background
bitmap. The editorial images are combined with animated SVG and CSS; movement
and information packets are not baked into those bitmaps.

Before publishing, confirm that `hello@mycellios.com` can receive email or update
the `EMAIL` constant in `src/Landing.tsx` with the final address.
