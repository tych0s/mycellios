#!/usr/bin/env bash
set -euo pipefail

source_repo=${1:?usage: sanitize-history.sh SOURCE_REPO SOURCE_REF DESTINATION}
source_ref=${2:?usage: sanitize-history.sh SOURCE_REPO SOURCE_REF DESTINATION}
destination=${3:?usage: sanitize-history.sh SOURCE_REPO SOURCE_REF DESTINATION}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [[ -e "$destination" ]]; then
  echo "destination already exists: $destination" >&2
  exit 2
fi

uv_bin=${UV_BIN:-/srv/jarvis-prod/projects/jarvis/.state/uv/uv}
if [[ ! -x "$uv_bin" ]]; then
  echo "uv is required; set UV_BIN to its executable path" >&2
  exit 3
fi
filter_repo=("$uv_bin" tool run --from 'git-filter-repo==2.47.0' git-filter-repo)

git init --quiet "$destination"
git -C "$destination" fetch --quiet --no-tags "$source_repo" "$source_ref:refs/heads/main"
git -C "$destination" checkout --quiet main

filter_args=()
while IFS= read -r path; do
  [[ -z "$path" || "$path" == \#* ]] && continue
  if [[ "$path" == *'*'* || "$path" == *'?'* || "$path" == *'['* ]]; then
    filter_args+=(--path-glob "$path")
  else
    filter_args+=(--path "$path")
  fi
done < "$script_dir/sanitization-paths.txt"

commit_callback='''
import re
commit.author_name = b"Mycellios Contributors"
commit.author_email = b"contributors@mycellios.com"
commit.committer_name = b"Mycellios Contributors"
commit.committer_email = b"contributors@mycellios.com"
message = commit.message.decode("utf-8", "replace")
if re.search(r"(?i)^(jarvis (publication|delivery)|publish: conversation snapshot|deploy: conversation snapshot)", message):
    message = "chore: integrate project changes\n"
message = re.sub(r"(?i)(?:origin/)?(?:agent|autopilot|checkpoint|claude|codex|jarvis-sync)/[A-Za-z0-9._/-]+", "private-branch", message)
message = re.sub(r"example/", "", message, flags=re.I)
message = re.sub(r"(?im)^Co-authored-by:.*(?:jarvis|claude|example).*(?:\n|$)", "", message)
message = re.sub(r"(?i)Mesh[-_ ]?LLM", "external runtime A", message)
message = re.sub(r"(?i)(?<![A-Za-z0-9_])EXO(?![A-Za-z0-9_])", "external runtime B", message)
message = re.sub(r"(?i)(?:leyten/shard|external/shard-runtime)", "external runtime C", message)
message = re.sub(r"(?i)(?<![A-Za-z0-9_])shard(?=\s+(?:1[,.]87|11[,.]5-50))", "external runtime C", message)
message = re.sub(r"(?i)Daniel", "ExampleUser", message)
message = re.sub(r"(?i)c0mpute", "external-compute-runtime", message)
message = re.sub(r"(?i)electron-zero", "native runtime boundary", message)
message = re.sub(r"(?i)sidecars?", "native helpers", message)
message = re.sub(r"(?i)Dokploy", "deployment platform", message)
message = re.sub(r"(?i)Infisical", "secret store", message)
message = re.sub(r"(?i)Jarvis", "automation", message)
message = re.sub(r"(?i)example", "project host", message)
message = re.sub(r"(?i)Salad", "GPU cloud provider", message)
message = re.sub(r"(?i)Nakshatra", "native stage prototype", message)
message = re.sub(r"(?i)llama[._ -]?cpp", "external GGUF runtime", message)
message = re.sub(r"(?i)Ollama|vLLM|Petals|AntSeed|Parallax", "external runtime", message)
commit.message = message.encode("utf-8")
'''

"${filter_repo[@]}" \
  --force \
  --source "$destination" \
  --target "$destination" \
  --refs refs/heads/main \
  --invert-paths \
  "${filter_args[@]}" \
  --replace-text "$script_dir/replacements.txt" \
  --commit-callback "$commit_callback"

# Four public product files once contained retired integrations. Remove their
# complete historical blobs, then restore only the sanitized source snapshot.
for public_file in .env.example README.md package.json src/core/config.ts; do
  mkdir -p "$destination/$(dirname "$public_file")"
  git --git-dir="$source_repo" show "$source_ref:$public_file" > "$destination/$public_file"
done
for public_doc in \
  docs/README.md \
  docs/ARCHITECTURE.md \
  docs/TWO_HOST_QUICKSTART.md \
  docs/STATUS_AND_EVIDENCE.md \
  docs/DEVELOPMENT.md \
  docs/REPOSITORY_STRUCTURE.md \
  docs/ROADMAP.md \
  docs/SECURITY.md \
  docs/openapi.yaml; do
  mkdir -p "$destination/$(dirname "$public_doc")"
  git --git-dir="$source_repo" show "$source_ref:$public_doc" > "$destination/$public_doc"
done
for public_workflow in \
  .github/workflows/public-quality.yml \
  .github/workflows/security.yml \
  .github/workflows/node-build.yml; do
  mkdir -p "$destination/$(dirname "$public_workflow")"
  git --git-dir="$source_repo" show "$source_ref:$public_workflow" > "$destination/$public_workflow"
done
for public_asset in \
  assets/mycellios-brand-board-v1.png \
  landing/public/assets/logos/logo.png \
  landing/src/rebrand/assets/mycellios-rebrand-hero.png \
  landing/src/rebrand/assets/mycellios-live-network.png; do
  mkdir -p "$destination/$(dirname "$public_asset")"
  git --git-dir="$source_repo" show "$source_ref:$public_asset" > "$destination/$public_asset"
done
"$uv_bin" run --python 3.12 python - "$destination" "$script_dir/replacements.txt" <<'PY'
from pathlib import Path
import re, sys
root = Path(sys.argv[1])
rules = []
for line in Path(sys.argv[2]).read_text().splitlines():
    if not line or line.startswith("#"):
        continue
    kind, expression = line.split(":", 1)
    source, replacement = expression.split("==>", 1)
    rules.append((kind, source, replacement))
for relative in (
    ".env.example",
    "README.md",
    "package.json",
    "src/core/config.ts",
):
    path = root / relative
    text = path.read_text()
    for kind, source, replacement in rules:
        text = text.replace(source, replacement) if kind == "literal" else re.sub(source, replacement, text)
    path.write_text(text)

for relative in (
    ".github/workflows/public-quality.yml",
    ".github/workflows/node-build.yml",
):
    path = root / relative
    path.write_text(path.read_text().replace(
        "      - run: npm run verify:native-boundary\n",
        "",
    ))

# Keep required images but remove non-rendering C2PA/CA metadata chunks.
for relative in (
    "assets/mycellios-brand-board-v1.png",
    "landing/public/assets/logos/logo.png",
    "landing/src/rebrand/assets/mycellios-rebrand-hero.png",
    "landing/src/rebrand/assets/mycellios-live-network.png",
):
    path = root / relative
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"not a PNG: {relative}")
    output = bytearray(data[:8])
    offset = 8
    while offset < len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        chunk_end = offset + 12 + length
        chunk_type = data[offset + 4:offset + 8]
        if chunk_type not in {b"caBX", b"caID", b"c2pa"}:
            output.extend(data[offset:chunk_end])
        offset = chunk_end
    path.write_bytes(output)
PY

install -m 0644 "$script_dir/GPL-3.0.txt" "$destination/LICENSE"
git -C "$destination" apply "$script_dir/post-filter.patch"
node "$script_dir/finalize-public-readme.mjs" "$destination"
NPM_CONFIG_IGNORE_SCRIPTS=true npm --prefix "$destination" install --package-lock-only \
  --save-exact pdfjs-dist@6.2.108
NPM_CONFIG_IGNORE_SCRIPTS=true npm --prefix "$destination" audit fix --package-lock-only
git -C "$destination" add -A
final_date=$(git -C "$destination" show -s --format=%aI HEAD)
GIT_AUTHOR_DATE="$final_date" GIT_COMMITTER_DATE="$final_date" git -C "$destination" \
  -c user.name='Mycellios Contributors' \
  -c user.email='contributors@mycellios.com' \
  commit --quiet -m 'chore: finalize public repository boundary'

git -C "$destination" remote remove origin 2>/dev/null || true
git -C "$destination" reflog expire --expire=now --all
git -C "$destination" gc --prune=now
git -C "$destination" fsck --full
printf '%s\n' "$source_ref" > "$destination/.git/opensource-source-ref"
echo "sanitized candidate: $(git -C "$destination" rev-parse HEAD)"
