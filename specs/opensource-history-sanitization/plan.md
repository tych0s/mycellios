# Plan

1. Congelar el SHA remoto y un inventario de refs, commits, tags y rutas
   sensibles.
2. Crear una copia local independiente para la reescritura; no operar sobre
   `agent/mycellios-opensource-d0184570a9` ni sobre `origin`.
3. Provisionar `git-filter-repo` en un entorno temporal sin añadir dependencias
   al proyecto.
4. Aplicar la política versionada de rutas: retirar workflows/configuración de
   la plataforma privada, specs operativas, correo/DNS, research competitivo,
   benchmarks raw, Salad, Supabase operativo, outputs y el experimento
   Mesh-LLM.
5. Ejecutar escaneo de secretos sobre commits y blobs antes y después del
   filtro; revisar también mensajes, autores, URLs y nombres de ramas/tags.
6. Revisar licencias y procedencia de imágenes, dependencias, modelos,
   benchmarks y documentación externa.
7. Ejecutar tests y comprobaciones en el snapshot saneado; registrar hashes
   nuevos y diferencias de alcance.
8. Crear una rama local `opensource/candidate-32da626` apuntando a la candidata
   verificada, sin push, y presentar únicamente las decisiones de licencia,
   assets e identidad. No publicar hasta recibir autorización final.
9. Ejecutar una auditoría pública independiente y, si aparecen referencias
   operativas residuales, ampliar la política reproducible y regenerar desde el
   mismo SHA fuente.
10. Reemplazar el repositorio público recién creado solo después de certificar
    la nueva candidata; recrearlo vacío evita conservar objetos de la candidata
    anterior.

## Riesgos y mitigaciones

- **Secreto histórico:** escaneo por contenido y revocación/rotación del
  proveedor antes de publicar.
- **Pérdida excesiva de historia:** mantener filtros mínimos y conservar los
  commits cuyo árbol siga siendo publicable.
- **Referencia interna residual:** escaneo de URLs, dominios, emails,
  workflows y mensajes de commit.
- **Objetos residuales en GitHub:** eliminar y recrear el repositorio público
  nuevo si una auditoría posterior exige otra reescritura; no hacer un simple
  force-push que deje objetos huérfanos accesibles por SHA.
- **Dependencia no licenciada:** inventario de terceros y bloqueo de la ruta
  afectada hasta obtener autorización o sustituirla.
