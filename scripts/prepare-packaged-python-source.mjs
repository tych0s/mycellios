import {
  existsSync,
  mkdirSync,
  rmSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import {
  NATIVE_PYTHON_PRODUCT_FILES,
  NATIVE_PYTHON_PRODUCT_MANIFEST,
  prepareNativePythonProductSource,
} from "./native-python-product-policy.mjs";

const workspace = resolve(import.meta.dirname, "..");
const sourceRoot = resolve(workspace, "python");
const buildRoot = resolve(workspace, "build");
const destinationRoot = resolve(buildRoot, "python");

if (dirname(destinationRoot) !== buildRoot) {
  throw new Error("Packaged Python source escaped the managed build directory.");
}
if (!existsSync(sourceRoot)) {
  throw new Error(`Mycellios Python source is missing: ${sourceRoot}.`);
}

rmSync(destinationRoot, { recursive: true, force: true });
mkdirSync(buildRoot, { recursive: true });
prepareNativePythonProductSource(sourceRoot, destinationRoot);

process.stdout.write(
  `Packaged Python source ready: ${NATIVE_PYTHON_PRODUCT_FILES.length} ` +
    `allowlisted files plus ${NATIVE_PYTHON_PRODUCT_MANIFEST}.\n`,
);
