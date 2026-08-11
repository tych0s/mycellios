import { spawn } from "node:child_process";
import { lstat, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { nodeEnrollmentBundleSchema } from "../contracts/node-control.js";

const enrollment = process.argv[2];
if (!enrollment) throw new Error("usage: macos-pairing-main <pairing.mycellios-enrollment>");
const enrollmentPath = resolve(enrollment);
if (!enrollmentPath.endsWith(".mycellios-enrollment")) throw new Error("node_pairing_file_extension_is_invalid");
const stats = await lstat(enrollmentPath);
if (!stats.isFile() || stats.isSymbolicLink() || stats.size > 4_096) throw new Error("node_pairing_file_is_unsafe");
nodeEnrollmentBundleSchema.parse(JSON.parse(await readFile(enrollmentPath, "utf8")) as unknown);
const product = "/Library/Application Support/Mycellios/Product";
const command = [
  `${product}/bin/node`, `${product}/app/node/install-main.js`, "--install-root", product, "--enrollment", enrollmentPath,
].map(shellQuote).join(" ");
const script = `do shell script ${appleScriptString(command)} with administrator privileges`;
const code = await run("/usr/bin/osascript", ["-e", script]);
if (code !== 0) throw new Error("node_pairing_privileged_install_failed");

function shellQuote(value: string): string { return `'${value.replaceAll("'", `'\\''`)}'`; }
function appleScriptString(value: string): string { return `"${value.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`; }
async function run(executable: string, arguments_: readonly string[]): Promise<number> {
  return await new Promise((accept, reject) => { const child = spawn(executable, [...arguments_], { shell: false, stdio: "ignore" });
    child.once("error", reject); child.once("close", (code_) => accept(code_ ?? -1)); });
}
