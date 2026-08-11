import { randomBytes } from "node:crypto";
import { constants } from "node:fs";
import {
  access,
  chmod,
  copyFile,
  mkdir,
  open,
  readFile,
  rename,
  rm,
} from "node:fs/promises";
import { dirname, resolve } from "node:path";
import {
  parseNodeConfiguration,
  type NodeConfiguration,
} from "../contracts/node-configuration.js";

export interface LoadedNodeConfiguration {
  config: NodeConfiguration;
  source: "primary" | "backup";
  migrated: boolean;
}

export class NodeConfigurationStore {
  readonly path: string;
  readonly backupPath: string;

  constructor(
    path: string,
    private readonly options: { allowLegacyMigration?: boolean } = {},
  ) {
    this.path = resolve(path);
    this.backupPath = `${this.path}.previous`;
  }

  async load(): Promise<LoadedNodeConfiguration> {
    const primary = await this.readCandidate(this.path);
    if (primary) return { ...primary, source: "primary" };
    const backup = await this.readCandidate(this.backupPath);
    if (!backup) throw new Error("mycellios_node_configuration_is_unreadable");
    return { ...backup, source: "backup" };
  }

  async save(input: unknown): Promise<NodeConfiguration> {
    const config = parseNodeConfiguration(input);
    await mkdir(dirname(this.path), { recursive: true, mode: 0o700 });
    const temporary = `${this.path}.tmp-${process.pid}-${randomBytes(8).toString("hex")}`;
    const backupTemporary = `${this.backupPath}.tmp-${process.pid}-${randomBytes(8).toString("hex")}`;
    try {
      await writeDurableJson(temporary, config);
      if (await exists(this.path)) {
        await copyFile(this.path, backupTemporary, constants.COPYFILE_EXCL);
        await chmod(backupTemporary, 0o600);
        await syncFile(backupTemporary);
        await rename(backupTemporary, this.backupPath);
      }
      await rename(temporary, this.path);
      await syncDirectory(dirname(this.path));
      return config;
    } finally {
      await Promise.all([
        rm(temporary, { force: true }),
        rm(backupTemporary, { force: true }),
      ]);
    }
  }

  async rollback(): Promise<NodeConfiguration> {
    const previous = await this.read(this.backupPath).catch(() => null);
    if (!previous) throw new Error("mycellios_node_configuration_backup_is_unreadable");
    const temporary = `${this.path}.rollback-${process.pid}-${randomBytes(8).toString("hex")}`;
    try {
      await writeDurableJson(temporary, previous.config);
      await rename(temporary, this.path);
      await syncDirectory(dirname(this.path));
      return previous.config;
    } finally {
      await rm(temporary, { force: true });
    }
  }

  private async read(path: string): Promise<Omit<LoadedNodeConfiguration, "source">> {
    const document = JSON.parse(await readFile(path, "utf8")) as unknown;
    const config = parseNodeConfiguration(document);
    const schema = typeof document === "object" && document !== null
      ? (document as Record<string, unknown>).schema
      : null;
    const migrated = schema !== config.schema;
    if (migrated && this.options.allowLegacyMigration !== true) {
      throw new Error("mycellios_node_legacy_configuration_migration_confirmation_required");
    }
    return { config, migrated };
  }

  private async readCandidate(path: string): Promise<Omit<LoadedNodeConfiguration, "source"> | null> {
    try {
      return await this.read(path);
    } catch (error) {
      if ((error as Error).message === "mycellios_node_legacy_configuration_migration_confirmation_required") {
        throw error;
      }
      return null;
    }
  }
}

async function writeDurableJson(path: string, value: unknown): Promise<void> {
  const file = await open(path, "wx", 0o600);
  try {
    await file.writeFile(`${JSON.stringify(value, null, 2)}\n`, "utf8");
    await file.sync();
  } finally {
    await file.close();
  }
}

async function syncFile(path: string): Promise<void> {
  const file = await open(path, "r");
  try { await file.sync(); } finally { await file.close(); }
}

async function syncDirectory(path: string): Promise<void> {
  const directory = await open(path, "r");
  try { await directory.sync(); } finally { await directory.close(); }
}

async function exists(path: string): Promise<boolean> {
  try { await access(path); return true; } catch { return false; }
}
