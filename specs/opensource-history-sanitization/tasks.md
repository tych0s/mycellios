# Tareas

- [x] Congelar inventario de refs y rutas históricas.
  Verificar: `git show-ref`, `git log --all` y `git ls-tree` reproducibles.
- [x] Crear copia local de trabajo para reescritura.
  Verificar: `git fsck --full` y comparación de refs con el original.
- [x] Clasificar rutas en conservar, excluir automáticamente y decisión humana.
  Verificar: lista versionada y revisión de cada categoría.
- [x] Ejecutar filtro histórico reproducible en la copia.
  Verificar: los patrones excluidos no aparecen en ningún commit saneado.
- [x] Escanear secretos y referencias internas antes/después.
  Verificar: informe sin críticos y hallazgos importantes explicados.
- [x] Añadir la licencia raíz elegida por el propietario.
  Verificar: texto canónico GPL v3 y SPDX `GPL-3.0-only`.
- [x] Resolver autorización de assets con el propietario.
  Verificar: el propietario confirmó los derechos de publicación de los 44
  assets auditados.
- [x] Aprobar la identidad colectiva del historial.
  Verificar: el propietario aprobó `Mycellios Contributors
  <contributors@mycellios.com>`.
- [x] Ejecutar pruebas del snapshot saneado.
  Verificar: typecheck/tests focalizados y ausencia de artefactos temporales.
- [x] Presentar informe para autorización de publicación.
  Verificar: no se ejecuta push ni cambio de visibilidad antes de la decisión.
- [x] Sustituir el `main` remoto y retirar refs ordinarias antiguas.
  Verificar: solo queda `refs/heads/main` en `df0c748a`; 0 tags y 0 releases.
- [x] Aislar las 95 refs internas de pull requests retenidas por GitHub.
  Verificar: permanecen únicamente en el repositorio organizativo `internal` y
  no se transfieren al repositorio público.
- [x] Publicar un repositorio limpio fuera de la organización.
  Verificar: `tych0s/mycellios` es público, `main` está en `36c49c76` y tiene
  0 tags, 0 releases y 0 pull requests.
- [x] Eliminar referencias operativas residuales detectadas por la auditoría
  pública reforzada.
  Verificar: 0 coincidencias históricas de Nodecodex, Jarvis, Dokploy e
  Infisical en contenido y metadatos.
- [x] Regenerar y certificar una candidata pública final.
  Verificar: clon anónimo, Gitleaks, refs, autores, paths y enlaces correctos.
- [x] Recrear `tych0s/mycellios` con la candidata final.
  Verificar: el repositorio público nuevo no conserva objetos de `df0c748a`.
- [x] Anonimizar `Daniel` y las alusiones externas ambiguas a `shard` en toda
  la historia, conservando sus usos técnicos.
  Verificar: 0 coincidencias de `dani`, `leyten/shard` y
  `external/shard-runtime`; tests y referencias técnicas de `shard` intactos.
- [x] Regenerar y publicar una candidata desde el `main` reconciliado vigente.
  Verificar: `e61f2ed355d6bb319630c105625c26a071b506e3`, build completo,
  1760 tests, Gitleaks sin hallazgos y clon anónimo del remoto público.
- [x] Eliminar residuos competitivos y operativos de contenido, rutas y mensajes.
  Verificar: cero coincidencias en todos los commits y nombres históricos.
- [x] Retirar assets sin uso con metadata C2PA externa y documentación rota.
  Verificar: ningún bloque `caBX` y cero enlaces locales rotos.
- [x] Recrear y certificar el repositorio público estricto.
  Verificar: clon anónimo limpio y objetos públicos anteriores inaccesibles.
