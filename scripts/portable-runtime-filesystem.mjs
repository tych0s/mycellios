import {
  readlinkSync,
  readdirSync,
  realpathSync,
  renameSync,
  statSync,
  symlinkSync,
  unlinkSync,
} from "node:fs";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
} from "node:path";

export function assertNoEscapingSymlinks(root) {
  const absoluteRoot = resolve(root);
  const canonicalRoot = realpathSync.native(absoluteRoot);

  forEachSymlink(absoluteRoot, (path) => {
    const target = readlinkSync(path);
    const resolvedTarget = resolve(dirname(path), target);

    let canonicalDirectTarget;
    let canonicalFinalTarget;
    try {
      // Checking the canonical parent as well as the final target rejects links
      // that temporarily leave the runtime through an intermediate directory.
      canonicalDirectTarget = join(
        realpathSync.native(dirname(resolvedTarget)),
        basename(resolvedTarget),
      );
      canonicalFinalTarget = realpathSync.native(path);
    } catch (error) {
      throw new Error(
        `Portable runtime contains a broken symlink: ${relative(absoluteRoot, path)} -> ${target}`,
        { cause: error },
      );
    }

    if (
      !isPathInside(canonicalRoot, canonicalDirectTarget) ||
      !isPathInside(canonicalRoot, canonicalFinalTarget)
    ) {
      throw new Error(
        `Portable runtime contains an escaping symlink: ${relative(absoluteRoot, path)} -> ${target}`,
      );
    }
  });
}

export function normalizeCopiedInternalAbsoluteSymlinks(copiedRoot, sourceRoot) {
  const absoluteCopiedRoot = resolve(copiedRoot);
  const canonicalSourceRoot = realpathSync.native(resolve(sourceRoot));
  let temporaryLinkCounter = 0;

  forEachSymlink(absoluteCopiedRoot, (path) => {
    const target = readlinkSync(path);
    if (!isAbsolute(target)) return;

    let canonicalTarget;
    let targetType;
    try {
      // At this point an absolute link in the verbatim copy still references
      // its certified source tree. Collapse it to the same physical entry in
      // the copied tree, then encode that destination as a relative link.
      canonicalTarget = realpathSync.native(path);
      targetType = statSync(path).isDirectory() ? "dir" : "file";
    } catch (error) {
      throw new Error(
        `Portable runtime contains a broken absolute symlink: ${relative(absoluteCopiedRoot, path)} -> ${target}`,
        { cause: error },
      );
    }

    if (!isPathInside(canonicalSourceRoot, canonicalTarget)) {
      throw new Error(
        `Portable runtime contains an escaping absolute symlink: ${relative(absoluteCopiedRoot, path)} -> ${target}`,
      );
    }

    const sourceRelativeTarget = relative(canonicalSourceRoot, canonicalTarget);
    const copiedTarget = resolve(absoluteCopiedRoot, sourceRelativeTarget);
    if (!isPathInside(absoluteCopiedRoot, copiedTarget)) {
      throw new Error(
        `Portable runtime absolute symlink could not be mapped safely: ${relative(absoluteCopiedRoot, path)} -> ${target}`,
      );
    }

    const portableTarget = relative(dirname(path), copiedTarget) || ".";
    const temporaryLink = `${path}.mycellios-relative-${process.pid}-${temporaryLinkCounter++}`;
    let temporaryLinkCreated = false;
    try {
      symlinkSync(portableTarget, temporaryLink, targetType);
      temporaryLinkCreated = true;
      unlinkSync(path);
      renameSync(temporaryLink, path);
    } catch (error) {
      if (temporaryLinkCreated) {
        try {
          unlinkSync(temporaryLink);
        } catch {
          // Keep the original error; cleanup is best effort.
        }
      }
      throw error;
    }
  });
}

export function samePath(left, right) {
  const normalize = (value) => {
    const absolute = resolve(String(value));
    let canonical = absolute;
    try {
      canonical = realpathSync.native(absolute);
    } catch {
      // Callers use this as a comparison predicate. A missing path should
      // compare by its normalized absolute spelling and fail naturally.
    }
    const normalized = canonical.replaceAll("\\", "/");
    return process.platform === "win32" ? normalized.toLowerCase() : normalized;
  };
  return normalize(left) === normalize(right);
}

function forEachSymlink(root, visitor) {
  const visit = (directory) => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        visitor(path);
      } else if (entry.isDirectory()) {
        visit(path);
      }
    }
  };
  visit(root);
}

function isPathInside(root, target) {
  const back = relative(root, target);
  return back === "" || (back !== ".." && !back.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`) && !isAbsolute(back));
}
