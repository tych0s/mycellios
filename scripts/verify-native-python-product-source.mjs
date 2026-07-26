import { resolve } from "node:path";
import {
  NATIVE_PYTHON_PRODUCT_FILES,
  assertNativePythonSourceClosure,
} from "./native-python-product-policy.mjs";

const workspace = resolve(import.meta.dirname, "..");
assertNativePythonSourceClosure(resolve(workspace, "python"));
process.stdout.write(
  `Native Python source closure verified: ${NATIVE_PYTHON_PRODUCT_FILES.length} allowlisted files.\n`,
);
