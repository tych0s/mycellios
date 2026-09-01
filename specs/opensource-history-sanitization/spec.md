# Sanitización del historial para publicación open source

## Objetivo

Preparar el historial Git de Mycellios para publicación pública conservando la
cronología y la evolución técnica, pero eliminando secretos, operaciones
privadas, infraestructura interna, datos de experimentos no publicables y
documentación cuya publicación no esté autorizada.

## Alcance

- Auditar todos los commits, tags y referencias alcanzables del repositorio.
- Reescribir una copia de trabajo del historial, sin sobrescribir la rama
  actual ni publicar remotamente.
- Aplicar una política reproducible de rutas y transformaciones.
- Ejecutar comprobaciones históricas de secretos, referencias internas,
  artefactos grandes y licencias.
- Entregar un informe de hallazgos importantes y una lista de decisiones
  pendientes antes de cualquier push o cambio de visibilidad.

## No objetivos

- No publicar, hacer público ni sobrescribir `origin` en esta fase.
- No revocar ni rotar credenciales sin identificar primero el proveedor y el
  alcance; cualquier secreto confirmado requerirá rotación posterior.
- No eliminar investigación o benchmarks solo por ser antiguos: se clasifican
  y se decide según sensibilidad, procedencia y reproducibilidad.

## Criterios de aceptación

1. La copia saneada conserva el orden temporal y la evolución técnica
   publicable.
2. Ningún secreto detectado queda en blobs, mensajes, tags o referencias del
   historial saneado.
3. No quedan workflows de despliegue privado ni credenciales/configuración de
   producción en el contenido público.
4. Cada exclusión relevante queda documentada con su motivo.
5. El repositorio original y sus referencias permanecen intactos durante la
   preparación.
6. La publicación queda bloqueada si persiste un hallazgo crítico o falta una
   decisión de propiedad/licencia.

## Snapshot congelado

- Fuente: `origin/main` en
  `32da626f891e216651dda2fbc857b7eaf5718d64` (496 commits alcanzables).
- La candidata se construye fuera del worktree compartido y no conserva un
  remoto con permisos de escritura.
- La reescritura debe poder repetirse con `sanitize-history.sh` y
  `sanitization-paths.txt`.

## Decisiones aplicadas

- Excluir del historial los artefactos de operación privada, publicación,
  despliegue, correo/DNS, experimentación competitiva y resultados raw.
- Anonimizar autores y committers históricos con una identidad colectiva de
  Mycellios.
- Normalizar mensajes de merges/agentes para retirar nombres de ramas y
  automatizaciones internas sin borrar la cronología técnica.
- Anonimizar nombres personales usados en fixtures y retirar alusiones al
  proyecto externo `shard`, conservando `shard` como término técnico.

## Decisiones del propietario

- Licencia open source definitiva: GNU GPL v3, variante `GPL-3.0-only`.
- Publicación autorizada de los 44 assets auditados.
- Identidad histórica colectiva autorizada: `Mycellios Contributors
  <contributors@mycellios.com>`.
- Repositorio público limpio: `tych0s/mycellios`; el repositorio organizativo
  permanece `internal` como archivo.

## Frontera pública reforzada

- No deben aparecer en ningún árbol o metadato histórico nombres, dominios,
  rutas o configuración de Nodecodex, Jarvis, Dokploy o Infisical.
- Los enlaces públicos deben apuntar a `tych0s/mycellios` o a endpoints públicos
  de Mycellios, nunca al repositorio organizativo archivado.
- La certificación final se ejecuta desde un clon anónimo del repositorio
  público y abarca Gitleaks, refs, autores, rutas excluidas y enlaces locales.
