# Informe de candidata open source

## Identidad

- Fuente congelada: `32da626f891e216651dda2fbc857b7eaf5718d64`.
- Candidata saneada final: `494b19ac08dbca0584b9ed1dd14ef6fae8120daf`.
- Rama local: `opensource/candidate-32da626`.
- Historial: 309 commits, todos con la identidad colectiva
  `Mycellios Contributors <contributors@mycellios.com>`.

## Verificación

- Reconstrucción reproducible: dos ejecuciones emitieron el mismo SHA.
- `git fsck --full`: correcto, sin objetos residuales alcanzables.
- Gitleaks 8.30.1: 244 commits con contenido escaneable, 0 hallazgos con la política de
  falsos positivos documentada en `.gitleaks.toml`.
- Referencias privadas/competitivas definidas por la política: 0 en los 309
  árboles históricos y 0 en metadatos de commits.
- Typecheck core, landing y mobile: correctos.
- Suite Vitest completa: 1747 tests correctos y 16 omitidos usando Python 3.12
  para el analizador AST.
- `npm audit`: 0 vulnerabilidades después de actualizar `pdfjs-dist` y las
  dependencias transitivas corregibles.
- Licencia: texto canónico GNU GPL v3, declarado como `GPL-3.0-only` en
  `package.json` y documentado en README; marcas excluidas expresamente.
- Documentación pública: 14 archivos Markdown auditados, 0 enlaces locales
  rotos; eliminado el aviso obsoleto de licencia pendiente.
- Assets: 44 archivos visuales, tipográficos o multimedia rastreados hasta su
  primera incorporación al historial, todos bajo `Daniel <daniel@nodecodex.io>`;
  no se encontraron autores externos ni metadatos descriptivos incrustados.
  La landing identifica dos escenas editoriales como generadas con IA.

## Autorizaciones del propietario

- El propietario confirmó que posee los derechos necesarios para publicar los
  44 assets auditados, incluidos los dos declarados como IA.
- El propietario aprobó la identidad histórica colectiva
  `Mycellios Contributors <contributors@mycellios.com>`.

No quedan bloqueos documentales o de derechos detectados por esta auditoría.
El propietario autorizó posteriormente el corte remoto. `main` se sustituyó por
la candidata saneada y se eliminaron 56 ramas auxiliares, 58 tags y 59 releases
del repositorio remoto. GitHub conserva 95 refs internas `refs/pull/*` que no se
pueden eliminar mediante la API normal y que todavía alcanzan objetos del
historial anterior.

El repositorio permanece con visibilidad `internal`: hacerlo público antes de
resolver esas refs expondría de nuevo parte del historial saneado. La regla
organizativa del branch por defecto fue restaurada en estado activo después del
force-push.

## Publicación limpia

Para evitar heredar las refs internas de GitHub, el propietario autorizó crear
un repositorio nuevo fuera de la organización. La candidata se publicó como
`tych0s/mycellios`, con visibilidad `public` y `main` en
`494b19ac08dbca0584b9ed1dd14ef6fae8120daf`.

Tras detectar referencias operativas residuales en una auditoría reforzada, el
repositorio público inicial se eliminó y recreó vacío para no conservar sus
objetos por SHA. La verificación remota y anónima final confirma una única rama
(`main`), 0 tags, 0 releases, 0 issues y 0 pull requests. El repositorio
`nodecodex-org/mycellios` permanece `internal` como archivo y sus refs de pull
request no se transfirieron.

Una revisión adicional retiró `Daniel` de los fixtures históricos y las
alusiones ambiguas al proyecto externo `shard`, conservando 424 usos actuales
del término técnico. La candidata se reconstruyó dos veces con el mismo SHA,
superó 1747 tests, los tres typechecks, Gitleaks y `npm audit`, y el repositorio
público se recreó de nuevo para no conservar el objeto anterior por SHA.

## Reconciliación posterior del 30-08-2026

Una publicación interna posterior había reincorporado documentación, benchmarks
y referencias personales/competitivas al `main` organizativo. Se regeneró la
candidata pública desde `47602457758f4cc943491191a8c0f83973e00233` usando la
misma política reproducible, actualizada para el árbol vigente.

- Candidata pública: `e61f2ed355d6bb319630c105625c26a071b506e3`.
- Historial alcanzable: 313 commits, una única identidad colectiva.
- Cero coincidencias históricas de `dani`, `leyten/shard`, `c0mpute`, Mesh LLM,
  Nodecodex, Jarvis, Dokploy e Infisical.
- Gitleaks 8.30.1: 248 commits con contenido, 0 hallazgos.
- Typechecks core, landing y mobile correctos; build completo correcto.
- Vitest: 257 archivos y 1760 tests correctos; 5 archivos y 16 tests físicos
  omitidos.
- Verificación remota anónima: `tych0s/mycellios` público, una rama (`main`),
  0 tags, 0 releases y 0 pull requests.

## Frontera pública estricta del 30-08-2026

Una segunda auditoría recorrió contenido, rutas y mensajes de todos los commits,
además de metadata binaria, dependencias y enlaces documentales. La política se
amplió para retirar adaptadores y experimentos externos, referencias de proveedor,
comparativas, rutas históricas residuales y assets sin uso. Los PNG necesarios se
conservaron sin sus bloques C2PA externos. Stripe y Supabase se mantienen solo como
integraciones funcionales del producto.

- Candidata reproducible: `7762a33f79125701a08c1730bc3db70a2d831a33`.
- Historial: 306 commits y una identidad colectiva.
- Cero patrones prohibidos en contenido, paths o mensajes de todos los commits.
- Cero bloques C2PA/Trufo y cero secretos según Gitleaks.
- `npm audit`: cero vulnerabilidades en el runtime publicado; `docs-site` se retiró
  porque su cadena Docusaurus mantenía 22 alertas altas, incluidas dependencias sin
  corrección disponible.
- Typecheck y build completos correctos; 1760 tests correctos y 15 físicos omitidos.
- Repositorio público recreado y verificado desde clon anónimo; los objetos públicos
  anteriores dejaron de ser recuperables por SHA.
- Secret scanning, push protection y actualizaciones de seguridad de Dependabot
  activados en GitHub.
