# Landing de mycellios

Sitio público independiente del cliente Electron, preparado para publicarse en
`https://mycellios.com`.

## Desarrollo

```powershell
npm run landing:dev
```

## Validación y build de producción

```powershell
npm run landing:typecheck
npm run landing:build
npm run landing:preview
```

La salida estática se genera en `landing-dist/`. Ese directorio es el que debe
publicar el proveedor de hosting. Las rutas públicas incluyen `robots.txt`,
`sitemap.xml` y el favicon de la marca.

La interfaz detecta el idioma del navegador, permite alternar entre español e
inglés y conserva la preferencia localmente. Los metadatos principales también
se actualizan con el idioma seleccionado.

## Activos visuales

Las escenas originales de micelio generadas para la web están en:

- `src/assets/mycelium-hero.jpg`
- `src/assets/mycelium-network.jpg`

Las imágenes se combinan con SVG y CSS animado; el movimiento y los paquetes de
información no están horneados en el bitmap.

Antes de publicar, confirma que `hello@mycellios.com` recibe correo o cambia la
constante `EMAIL` de `src/Landing.tsx` por la dirección definitiva.
