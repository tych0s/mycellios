import type {
  CompletionMetrics,
  JobStatus,
  ScheduledRoute,
  WorkerCapabilities,
  WorkerRegistration,
  WorkerStatus,
} from "../contracts/types.js";
import type { RemoteDiagnosticEvent } from "../contracts/remote-diagnostics.js";
import type { BenchmarkRun } from "../benchlab/types.js";
import type {
  FederatedNetworkId,
  FederatedNetworkSettings,
  FederatedRouteAttempt,
  FederationSettings,
  ManagedRental,
} from "../contracts/federation.js";
import { newId } from "../core/ids.js";
import {
  ASSISTANT_SETTINGS_ID,
  DEFAULT_SUPPORT_ASSISTANT_SETTINGS,
  type SupportAssistantSettings,
} from "../support/assistant.js";
import { MeshDatabase } from "./database.js";

function startOfUtcDay(timestamp: number): number {
  const date = new Date(timestamp);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

export interface StoredWorker {
  id: string;
  status: WorkerStatus;
  capabilities: WorkerCapabilities;
  reliability: number;
  jobsCompleted: number;
  lastSeenAt: number;
  identityKind: "device" | "cell" | null;
  identityId: string | null;
}

export interface StoredJob {
  id: string;
  sessionId: string;
  model: string;
  workloadClass: string;
  status: JobStatus;
  workerId: string | null;
  deploymentId: string | null;
  modelDigest: string | null;
  leaseId: string | null;
  inputTokens: number;
  outputTokens: number;
  failureCode: string | null;
  deadlineAt: number;
  createdAt: number;
  updatedAt: number;
}

interface WorkerRow {
  id: string;
  status: WorkerStatus;
  capabilities_json: string;
  reliability: number;
  jobs_completed: number;
  last_seen_at: number;
  identity_kind: "device" | "cell" | null;
  identity_id: string | null;
}

interface JobRow {
  id: string;
  session_id: string;
  model: string;
  workload_class: string;
  status: JobStatus;
  worker_id: string | null;
  deployment_id: string | null;
  model_digest: string | null;
  lease_id: string | null;
  input_tokens: number;
  output_tokens: number;
  failure_code: string | null;
  deadline_at: number;
  created_at: number;
  updated_at: number;
}

export interface StoredRequestedModel {
  id: string;
  source: string;
  revision: string | null;
  contextTokens: number;
  minimumNodes: number;
  autoActivate: boolean;
  profile: Record<string, unknown> | null;
  profileError: string | null;
  activationRequestedAt: number | null;
  activationError: string | null;
  createdAt: number;
  updatedAt: number;
}

export interface StoredInferenceMessage {
  id: string;
  conversationId: string;
  jobId: string | null;
  role: "system" | "developer" | "user" | "assistant" | "tool";
  content: string;
  status: "pending" | "completed" | "failed";
  inputTokens: number | null;
  outputTokens: number | null;
  routeClass: string | null;
  latencyMs: number | null;
  metadata: Record<string, unknown>;
  createdAt: number;
}

export interface StoredActivationEvent {
  phase: string;
  message: string;
  at: string;
  state: "running" | "completed" | "failed";
  nodeId?: string;
  processId?: string;
  device?: string;
  details?: readonly string[];
}

export interface StoredNetworkTelemetrySample {
  capturedAt: number;
  registeredNodes: number;
  connectedNodes: number;
  onlineNodes: number;
  browserNodes: number;
  activeModels: number;
  modelReplicas: number;
  modelPipelines: number;
  offeredVramMb: number;
  freeVramMb: number;
  inflightJobs: number;
  runningJobs: number;
  completedJobs: number;
}

export interface StoredDiagnosticEvent extends RemoteDiagnosticEvent {
  receivedAt: string;
}

export interface DiagnosticEventQuery {
  limit?: number;
  level?: RemoteDiagnosticEvent["level"];
  sourceId?: string;
  since?: string;
}

interface RequestedModelRow {
  id: string;
  source: string;
  revision: string | null;
  context_tokens: number;
  minimum_nodes: number;
  auto_activate: number;
  profile_json: string | null;
  profile_error: string | null;
  activation_requested_at: number | null;
  activation_error: string | null;
  created_at: number;
  updated_at: number;
}

export class MeshStore {
  constructor(readonly database: MeshDatabase) {}

  getFederationSettings(defaultEnabled = false): FederationSettings {
    const row = this.database.raw.prepare(
      "SELECT * FROM federation_settings WHERE id = 'global'",
    ).get() as {
      enabled: number;
      daily_budget_usd: number;
      monthly_budget_usd: number;
      autoscaling_enabled: number;
      max_rentals: number;
      updated_at: number;
    } | undefined;
    if (row) {
      return {
        enabled: Boolean(row.enabled),
        dailyBudgetUsd: Number(row.daily_budget_usd),
        monthlyBudgetUsd: Number(row.monthly_budget_usd),
        autoscalingEnabled: Boolean(row.autoscaling_enabled),
        maxRentals: Number(row.max_rentals),
        updatedAt: Number(row.updated_at),
      };
    }
    return this.saveFederationSettings({
      enabled: defaultEnabled,
      dailyBudgetUsd: 0,
      monthlyBudgetUsd: 0,
      autoscalingEnabled: false,
      maxRentals: 4,
    });
  }

  saveFederationSettings(
    input: Omit<FederationSettings, "updatedAt">,
  ): FederationSettings {
    const updatedAt = Date.now();
    this.database.raw.prepare(
      `INSERT INTO federation_settings(
         id, enabled, daily_budget_usd, monthly_budget_usd,
         autoscaling_enabled, max_rentals, updated_at
       ) VALUES ('global', ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         enabled = excluded.enabled,
         daily_budget_usd = excluded.daily_budget_usd,
         monthly_budget_usd = excluded.monthly_budget_usd,
         autoscaling_enabled = excluded.autoscaling_enabled,
         max_rentals = excluded.max_rentals,
         updated_at = excluded.updated_at`,
    ).run(
      input.enabled ? 1 : 0,
      input.dailyBudgetUsd,
      input.monthlyBudgetUsd,
      input.autoscalingEnabled ? 1 : 0,
      input.maxRentals,
      updatedAt,
    );
    this.queueFederationSettings();
    return { ...input, updatedAt };
  }

  getFederatedNetworkSettings(
    id: FederatedNetworkId,
    defaults: Omit<FederatedNetworkSettings, "id" | "updatedAt">,
  ): FederatedNetworkSettings {
    const row = this.database.raw.prepare(
      "SELECT * FROM federated_network_settings WHERE id = ?",
    ).get(id) as {
      enabled: number;
      priority: number;
      daily_budget_usd: number;
      monthly_budget_usd: number;
      updated_at: number;
    } | undefined;
    if (row) {
      return {
        id,
        enabled: Boolean(row.enabled),
        priority: Number(row.priority),
        dailyBudgetUsd: Number(row.daily_budget_usd),
        monthlyBudgetUsd: Number(row.monthly_budget_usd),
        updatedAt: Number(row.updated_at),
      };
    }
    return this.saveFederatedNetworkSettings({ id, ...defaults });
  }

  listFederatedNetworkSettings(): FederatedNetworkSettings[] {
    const rows = this.database.raw.prepare(
      "SELECT * FROM federated_network_settings ORDER BY priority, id",
    ).all() as unknown as Array<{
      id: FederatedNetworkId;
      enabled: number;
      priority: number;
      daily_budget_usd: number;
      monthly_budget_usd: number;
      updated_at: number;
    }>;
    return rows.map((row) => ({
      id: row.id,
      enabled: Boolean(row.enabled),
      priority: Number(row.priority),
      dailyBudgetUsd: Number(row.daily_budget_usd),
      monthlyBudgetUsd: Number(row.monthly_budget_usd),
      updatedAt: Number(row.updated_at),
    }));
  }

  saveFederatedNetworkSettings(
    input: Omit<FederatedNetworkSettings, "updatedAt">,
  ): FederatedNetworkSettings {
    const updatedAt = Date.now();
    this.database.raw.prepare(
      `INSERT INTO federated_network_settings(
         id, enabled, priority, daily_budget_usd, monthly_budget_usd, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         enabled = excluded.enabled,
         priority = excluded.priority,
         daily_budget_usd = excluded.daily_budget_usd,
         monthly_budget_usd = excluded.monthly_budget_usd,
         updated_at = excluded.updated_at`,
    ).run(
      input.id,
      input.enabled ? 1 : 0,
      input.priority,
      input.dailyBudgetUsd,
      input.monthlyBudgetUsd,
      updatedAt,
    );
    this.queueFederatedNetworkSettings(input.id);
    return { ...input, updatedAt };
  }

  reserveProviderSpend(input: {
    id: string;
    provider: FederatedNetworkId;
    requestId: string;
    maximumUsd: number;
    dailyBudgetUsd: number;
    monthlyBudgetUsd: number;
    now?: number;
  }): boolean {
    if (input.maximumUsd <= 0) return true;
    return this.database.transaction(() => {
      const now = input.now ?? Date.now();
      const dayStart = startOfUtcDay(now);
      const monthStart = Date.UTC(
        new Date(now).getUTCFullYear(),
        new Date(now).getUTCMonth(),
        1,
      );
      const sumSince = (since: number) => {
        const row = this.database.raw.prepare(
          `SELECT COALESCE(SUM(
             CASE WHEN status = 'reserved' THEN reserved_usd
                  WHEN status = 'reconciled' THEN COALESCE(actual_usd, reserved_usd)
                  ELSE 0 END
           ), 0) AS spent
           FROM provider_spend_reservations
           WHERE provider = ? AND created_at >= ?`,
        ).get(input.provider, since) as { spent: number };
        return Number(row.spent);
      };
      if (
        input.dailyBudgetUsd <= 0
        || input.monthlyBudgetUsd <= 0
        || sumSince(dayStart) + input.maximumUsd > input.dailyBudgetUsd
        || sumSince(monthStart) + input.maximumUsd > input.monthlyBudgetUsd
      ) return false;
      this.database.raw.prepare(
        `INSERT INTO provider_spend_reservations(
           id, provider, request_id, reserved_usd, status, created_at
         ) VALUES (?, ?, ?, ?, 'reserved', ?)`,
      ).run(input.id, input.provider, input.requestId, input.maximumUsd, now);
      this.queueFederationTableRow("provider_spend_reservations", input.id);
      return true;
    });
  }

  reconcileProviderSpend(id: string, actualUsd: number): void {
    this.database.raw.prepare(
      `UPDATE provider_spend_reservations
       SET actual_usd = ?, status = 'reconciled', reconciled_at = ?
       WHERE id = ? AND status = 'reserved'`,
    ).run(Math.max(0, actualUsd), Date.now(), id);
    this.queueFederationTableRow("provider_spend_reservations", id);
  }

  releaseProviderSpend(id: string): void {
    this.database.raw.prepare(
      `UPDATE provider_spend_reservations
       SET status = 'released', reconciled_at = ?
       WHERE id = ? AND status = 'reserved'`,
    ).run(Date.now(), id);
    this.queueFederationTableRow("provider_spend_reservations", id);
  }

  providerReservedForRequest(requestId: string): number {
    const row = this.database.raw.prepare(
      `SELECT COALESCE(SUM(reserved_usd), 0) AS reserved
       FROM provider_spend_reservations
       WHERE request_id = ? AND status = 'reserved'`,
    ).get(requestId) as { reserved: number };
    return Number(row.reserved);
  }

  reconcileProviderSpendForRequest(requestId: string, actualUsd: number): void {
    this.database.transaction(() => {
      const rows = this.database.raw.prepare(
        `SELECT id FROM provider_spend_reservations
         WHERE request_id = ? AND status = 'reserved'
         ORDER BY created_at, id`,
      ).all(requestId) as unknown as Array<{ id: string }>;
      rows.forEach((row, index) => {
        this.reconcileProviderSpend(row.id, index === 0 ? Math.max(0, actualUsd) : 0);
      });
    });
  }

  providerSpend(provider: FederatedNetworkId, now = Date.now()): {
    todayUsd: number;
    monthUsd: number;
  } {
    const total = (since: number) => {
      const row = this.database.raw.prepare(
        `SELECT COALESCE(SUM(
           CASE WHEN status = 'reserved' THEN reserved_usd
                WHEN status = 'reconciled' THEN COALESCE(actual_usd, reserved_usd)
                ELSE 0 END
         ), 0) AS spent
         FROM provider_spend_reservations
         WHERE provider = ? AND created_at >= ?`,
      ).get(provider, since) as { spent: number };
      return Number(row.spent);
    };
    const date = new Date(now);
    return {
      todayUsd: total(startOfUtcDay(now)),
      monthUsd: total(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), 1)),
    };
  }

  createFederatedRouteAttempt(attempt: FederatedRouteAttempt): void {
    this.database.raw.prepare(
      `INSERT INTO federated_route_attempts(
         id, request_id, provider, canonical_model, external_model, route_kind,
         started_at, first_token_at, completed_at, input_tokens, output_tokens,
         reserved_cost_usd, actual_cost_usd, result, fallback_reason, failure_code
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      attempt.id,
      attempt.requestId,
      attempt.provider,
      attempt.canonicalModel,
      attempt.externalModel,
      attempt.routeKind,
      attempt.startedAt,
      attempt.firstTokenAt,
      attempt.completedAt,
      attempt.inputTokens,
      attempt.outputTokens,
      attempt.reservedCostUsd,
      attempt.actualCostUsd,
      attempt.result,
      attempt.fallbackReason,
      attempt.failureCode,
    );
    this.queueFederationTableRow("federated_route_attempts", attempt.id);
  }

  updateFederatedRouteAttempt(
    id: string,
    input: Partial<Pick<
      FederatedRouteAttempt,
      "firstTokenAt" | "completedAt" | "inputTokens" | "outputTokens"
      | "actualCostUsd" | "result" | "fallbackReason" | "failureCode"
    >>,
  ): void {
    const entries = Object.entries({
      first_token_at: input.firstTokenAt,
      completed_at: input.completedAt,
      input_tokens: input.inputTokens,
      output_tokens: input.outputTokens,
      actual_cost_usd: input.actualCostUsd,
      result: input.result,
      fallback_reason: input.fallbackReason,
      failure_code: input.failureCode,
    }).filter((entry) => entry[1] !== undefined);
    if (entries.length === 0) return;
    this.database.raw.prepare(
      `UPDATE federated_route_attempts
       SET ${entries.map(([column]) => `${column} = ?`).join(", ")}
       WHERE id = ?`,
    ).run(...entries.map(([, value]) => value ?? null), id);
    this.queueFederationTableRow("federated_route_attempts", id);
  }

  listManagedRentals(): ManagedRental[] {
    const rows = this.database.raw.prepare(
      "SELECT * FROM managed_rentals ORDER BY created_at DESC",
    ).all() as unknown as Array<Record<string, unknown>>;
    return rows.map((row) => ({
      id: String(row.id),
      provider: row.provider as ManagedRental["provider"],
      state: row.state as ManagedRental["state"],
      image: String(row.image),
      requestedHardware: JSON.parse(String(row.hardware_json)) as ManagedRental["requestedHardware"],
      workerId: row.worker_id === null ? null : String(row.worker_id),
      reservedCostUsd: Number(row.reserved_cost_usd),
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
      drainStartedAt: row.drain_started_at === null ? null : Number(row.drain_started_at),
      stoppedAt: row.stopped_at === null ? null : Number(row.stopped_at),
      lastError: row.last_error === null ? null : String(row.last_error),
    }));
  }

  saveManagedRental(input: ManagedRental & {
    externalId: string;
    credentialIdentityId: string | null;
    labels: Record<string, string>;
  }): void {
    this.database.raw.prepare(
      `INSERT INTO managed_rentals(
         id, provider, external_id, state, image, hardware_json, worker_id,
         credential_identity_id, reserved_cost_usd, labels_json, created_at,
         updated_at, drain_started_at, stopped_at, last_error
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         state = excluded.state,
         worker_id = excluded.worker_id,
         reserved_cost_usd = excluded.reserved_cost_usd,
         updated_at = excluded.updated_at,
         drain_started_at = excluded.drain_started_at,
         stopped_at = excluded.stopped_at,
         last_error = excluded.last_error`,
    ).run(
      input.id,
      input.provider,
      input.externalId,
      input.state,
      input.image,
      JSON.stringify(input.requestedHardware),
      input.workerId,
      input.credentialIdentityId,
      input.reservedCostUsd,
      JSON.stringify(input.labels),
      input.createdAt,
      input.updatedAt,
      input.drainStartedAt,
      input.stoppedAt,
      input.lastError,
    );
    this.queueFederationTableRow("managed_rentals", input.id);
  }

  getManagedRentalPrivate(id: string): (ManagedRental & {
    externalId: string;
    credentialIdentityId: string | null;
    labels: Record<string, string>;
  }) | null {
    const row = this.database.raw.prepare(
      "SELECT * FROM managed_rentals WHERE id = ?",
    ).get(id) as Record<string, unknown> | undefined;
    if (!row) return null;
    const publicRental = this.listManagedRentals().find((rental) => rental.id === id);
    if (!publicRental) return null;
    return {
      ...publicRental,
      externalId: String(row.external_id),
      credentialIdentityId: row.credential_identity_id === null
        ? null
        : String(row.credential_identity_id),
      labels: JSON.parse(String(row.labels_json)) as Record<string, string>,
    };
  }

  getSupportAssistantSettings(): SupportAssistantSettings {
    const row = this.database.raw.prepare(
      `SELECT enabled, model_id, system_prompt, welcome_message, suggestions_json,
              max_output_tokens, temperature, allow_device_control, updated_at
       FROM assistant_settings
       WHERE id = ?`,
    ).get(ASSISTANT_SETTINGS_ID) as {
      enabled: number;
      model_id: string | null;
      system_prompt: string;
      welcome_message: string;
      suggestions_json: string;
      max_output_tokens: number;
      temperature: number;
      allow_device_control: number;
      updated_at: number;
    } | undefined;
    if (!row) return { ...DEFAULT_SUPPORT_ASSISTANT_SETTINGS };
    let suggestions = DEFAULT_SUPPORT_ASSISTANT_SETTINGS.suggestions;
    try {
      const parsed = JSON.parse(row.suggestions_json) as unknown;
      if (Array.isArray(parsed) && parsed.every((item) => typeof item === "string")) {
        suggestions = parsed;
      }
    } catch {
      // Keep the safe defaults if a manually edited row is malformed.
    }
    return {
      enabled: Boolean(row.enabled),
      modelId: row.model_id,
      systemPrompt: row.system_prompt,
      welcomeMessage: row.welcome_message,
      suggestions,
      maxOutputTokens: Number(row.max_output_tokens),
      temperature: Number(row.temperature),
      allowDeviceControl: Boolean(row.allow_device_control),
      updatedAt: Number(row.updated_at),
    };
  }

  saveSupportAssistantSettings(
    input: Omit<SupportAssistantSettings, "updatedAt">,
  ): SupportAssistantSettings {
    const updatedAt = Date.now();
    this.database.raw.prepare(
      `INSERT INTO assistant_settings(
         id, enabled, model_id, system_prompt, welcome_message, suggestions_json,
         max_output_tokens, temperature, allow_device_control, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         enabled = excluded.enabled,
         model_id = excluded.model_id,
         system_prompt = excluded.system_prompt,
         welcome_message = excluded.welcome_message,
         suggestions_json = excluded.suggestions_json,
         max_output_tokens = excluded.max_output_tokens,
         temperature = excluded.temperature,
         allow_device_control = excluded.allow_device_control,
         updated_at = excluded.updated_at`,
    ).run(
      ASSISTANT_SETTINGS_ID,
      input.enabled ? 1 : 0,
      input.modelId,
      input.systemPrompt,
      input.welcomeMessage,
      JSON.stringify(input.suggestions),
      input.maxOutputTokens,
      input.temperature,
      input.allowDeviceControl ? 1 : 0,
      updatedAt,
    );
    return { ...input, updatedAt };
  }

  registerWorker(registration: WorkerRegistration): StoredWorker {
    return this.database.transaction(() => {
      const now = Date.now();
      const identity = registration.identity;
      const existing = identity
        ? this.database.raw
            .prepare("SELECT id FROM workers WHERE identity_kind = ? AND identity_id = ?")
            .get(identity.kind, identity.id) as { id: string } | undefined
        : undefined;
      if (existing) {
        this.database.raw.prepare(
          `UPDATE workers
           SET capabilities_json = ?, last_seen_at = ?, updated_at = ?,
                deregistered = 0
           WHERE id = ?`,
        ).run(JSON.stringify(registration.capabilities), now, now, existing.id);
        this.queueWorker(existing.id);
        return this.getWorker(existing.id)!;
      }

      const workerId = newId("wrk");
      this.database.raw
        .prepare(
          `INSERT INTO workers(
             id, status, capabilities_json, last_seen_at, created_at, updated_at,
             identity_kind, identity_id
           ) VALUES (?, 'offline', ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          workerId,
          JSON.stringify(registration.capabilities),
          now,
          now,
          now,
          identity?.kind ?? null,
          identity?.id ?? null,
        );
      this.queueWorker(workerId);
      return this.getWorker(workerId)!;
    });
  }

  getWorker(workerId: string): StoredWorker | null {
    const row = this.database.raw
      .prepare("SELECT * FROM workers WHERE id = ? AND deregistered = 0")
      .get(workerId) as
      | WorkerRow
      | undefined;
    return row ? this.mapWorker(row) : null;
  }

  listWorkers(): StoredWorker[] {
    const rows = this.database.raw
      .prepare("SELECT * FROM workers WHERE deregistered = 0")
      .all() as unknown as WorkerRow[];
    return rows.map((row) => this.mapWorker(row));
  }

  listSchedulableWorkers(now = Date.now(), staleAfterMs = 15_000): StoredWorker[] {
    const rows = this.database.raw
      .prepare(
        "SELECT * FROM workers WHERE deregistered = 0 AND status = 'online' AND last_seen_at >= ?",
      )
      .all(now - staleAfterMs) as unknown as WorkerRow[];
    return rows.map((row) => this.mapWorker(row));
  }

  updateWorkerHeartbeat(
    workerId: string,
    capabilities: WorkerCapabilities,
    status: WorkerStatus,
  ): void {
    this.database.transaction(() => {
      const now = Date.now();
      this.database.raw
        .prepare(
          `UPDATE workers
           SET capabilities_json = ?, status = ?, last_seen_at = ?, updated_at = ?
           WHERE id = ?`,
        )
        .run(JSON.stringify(capabilities), status, now, now, workerId);
      this.queueWorker(workerId);
    });
  }

  setWorkerStatus(workerId: string, status: WorkerStatus): void {
    this.database.transaction(() => {
      this.database.raw
        .prepare("UPDATE workers SET status = ?, updated_at = ? WHERE id = ?")
        .run(status, Date.now(), workerId);
      this.queueWorker(workerId);
    });
  }

  deregisterWorker(workerId: string): boolean {
    return this.database.transaction(() => {
      const result = this.database.raw
        .prepare(
          `UPDATE workers
           SET deregistered = 1, status = 'offline', updated_at = ?
           WHERE id = ? AND deregistered = 0`,
        )
        .run(Date.now(), workerId);
      if (Number(result.changes) === 1) this.queueWorker(workerId);
      return Number(result.changes) === 1;
    });
  }

  deregisterOfflineWorkers(): number {
    return this.database.transaction(() => {
      const rows = this.database.raw.prepare(
        "SELECT id FROM workers WHERE deregistered = 0 AND status IN ('offline', 'suspect')",
      ).all() as unknown as Array<{ id: string }>;
      const result = this.database.raw
        .prepare(
          `UPDATE workers
           SET deregistered = 1, updated_at = ?
           WHERE deregistered = 0 AND status IN ('offline', 'suspect')`,
        )
        .run(Date.now());
      for (const row of rows) this.queueWorker(row.id);
      return Number(result.changes);
    });
  }

  upsertRequestedModel(input: {
    id: string;
    source: string;
    revision: string | null;
    contextTokens: number;
    minimumNodes: number;
    autoActivate: boolean;
  }): StoredRequestedModel {
    return this.database.transaction(() => {
      const now = Date.now();
      this.database.raw.prepare(
        `INSERT INTO requested_models(
           id, source, revision, context_tokens, minimum_nodes, auto_activate,
           created_at, updated_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           source = excluded.source,
           revision = excluded.revision,
           context_tokens = excluded.context_tokens,
           minimum_nodes = excluded.minimum_nodes,
           auto_activate = excluded.auto_activate,
           profile_json = NULL,
           profile_error = NULL,
           activation_requested_at = NULL,
           activation_error = NULL,
           updated_at = excluded.updated_at`,
      ).run(
        input.id,
        input.source,
        input.revision,
        input.contextTokens,
        input.minimumNodes,
        input.autoActivate ? 1 : 0,
        now,
        now,
      );
      this.queueRequestedModel(input.id);
      return this.getRequestedModel(input.id)!;
    });
  }

  getRequestedModel(id: string): StoredRequestedModel | null {
    const row = this.database.raw.prepare("SELECT * FROM requested_models WHERE id = ?").get(id) as
      | RequestedModelRow
      | undefined;
    return row ? this.mapRequestedModel(row) : null;
  }

  listRequestedModels(): StoredRequestedModel[] {
    const rows = this.database.raw
      .prepare("SELECT * FROM requested_models ORDER BY created_at DESC")
      .all() as unknown as RequestedModelRow[];
    return rows.map((row) => this.mapRequestedModel(row));
  }

  setRequestedModelProfile(
    id: string,
    profile: Record<string, unknown> | null,
    error: string | null,
  ): void {
    this.database.transaction(() => {
      this.database.raw.prepare(
        `UPDATE requested_models
         SET profile_json = ?, profile_error = ?, activation_requested_at = NULL,
             activation_error = NULL, updated_at = ?
         WHERE id = ?`,
      ).run(profile ? JSON.stringify(profile) : null, error, Date.now(), id);
      this.queueRequestedModel(id);
    });
  }

  setRequestedModelActivation(id: string, requested: boolean): void {
    this.database.transaction(() => {
      const now = Date.now();
      if (requested) {
        this.database.raw.prepare(
          `UPDATE requested_models
           SET activation_requested_at = ?, activation_error = NULL, updated_at = ?
           WHERE id = ?`,
        ).run(now, now, id);
      } else {
        this.database.raw.prepare(
          `UPDATE requested_models
           SET activation_requested_at = NULL, updated_at = ?
           WHERE id = ?`,
        ).run(now, id);
      }
      this.queueRequestedModel(id);
    });
  }

  setRequestedModelActivationError(id: string, error: string): void {
    this.database.transaction(() => {
      this.database.raw.prepare(
        `UPDATE requested_models
         SET activation_requested_at = NULL, activation_error = ?, updated_at = ?
         WHERE id = ?`,
      ).run(error, Date.now(), id);
      this.queueRequestedModel(id);
    });
  }

  clearRequestedModelActivationError(id: string): void {
    this.database.transaction(() => {
      this.database.raw.prepare(
        `UPDATE requested_models
         SET activation_requested_at = NULL, activation_error = NULL, updated_at = ?
         WHERE id = ?`,
      ).run(Date.now(), id);
      this.queueRequestedModel(id);
    });
  }

  removeRequestedModel(id: string): boolean {
    return this.database.transaction(() => {
      const removed = Number(
        this.database.raw.prepare("DELETE FROM requested_models WHERE id = ?").run(id).changes,
      ) === 1;
      if (removed) this.database.enqueueRemoteChange("requested_models", id, "delete", null);
      return removed;
    });
  }

  markStaleWorkers(now = Date.now()): { suspect: number; offline: number } {
    return this.database.transaction(() => {
      const candidates = this.database.raw.prepare(
        `SELECT id FROM workers
         WHERE (status = 'online' AND last_seen_at < ?)
            OR (status IN ('online', 'suspect') AND last_seen_at < ?)`,
      ).all(now - 10_000, now - 15_000) as unknown as Array<{ id: string }>;
      const suspect = this.database.raw
        .prepare(
          `UPDATE workers SET status = 'suspect', updated_at = ?
           WHERE status = 'online' AND last_seen_at < ?`,
        )
        .run(now, now - 10_000).changes;
      const offline = this.database.raw
        .prepare(
          `UPDATE workers SET status = 'offline', updated_at = ?
           WHERE status IN ('online', 'suspect') AND last_seen_at < ?`,
        )
        .run(now, now - 15_000).changes;
      for (const candidate of candidates) this.queueWorker(candidate.id);
      return { suspect: Number(suspect), offline: Number(offline) };
    });
  }

  createJob(input: {
    id: string;
    sessionId: string;
    model: string;
    workloadClass: string;
    deadlineAt: number;
  }): StoredJob {
    return this.database.transaction(() => {
      const now = Date.now();
      this.database.raw
        .prepare(
          `INSERT INTO jobs(
             id, session_id, model, workload_class, status, deadline_at, created_at, updated_at
           ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)`,
        )
        .run(
          input.id,
          input.sessionId,
          input.model,
          input.workloadClass,
          input.deadlineAt,
          now,
          now,
        );
      this.queueJob(input.id);
      return this.getJob(input.id)!;
    });
  }

  getIdempotentJob(idempotencyKey: string): { jobId: string; requestHash: string } | null {
    const row = this.database.raw
      .prepare("SELECT job_id, request_hash FROM idempotency_keys WHERE idempotency_key = ?")
      .get(idempotencyKey) as { job_id: string; request_hash: string } | undefined;
    return row ? { jobId: row.job_id, requestHash: row.request_hash } : null;
  }

  bindIdempotencyKey(idempotencyKey: string, requestHash: string, jobId: string): void {
    this.database.transaction(() => {
      const createdAt = Date.now();
      this.database.raw
        .prepare(
          `INSERT INTO idempotency_keys(idempotency_key, request_hash, job_id, created_at)
           VALUES (?, ?, ?, ?)`,
        )
        .run(idempotencyKey, requestHash, jobId, createdAt);
      this.database.enqueueRemoteChange("idempotency_keys", idempotencyKey, "upsert", {
        idempotency_key: idempotencyKey,
        request_hash: requestHash,
        job_id: jobId,
        created_at: createdAt,
      });
    });
  }

  getJob(jobId: string): StoredJob | null {
    const row = this.database.raw.prepare("SELECT * FROM jobs WHERE id = ?").get(jobId) as
      | JobRow
      | undefined;
    return row ? this.mapJob(row) : null;
  }

  setJobRoute(jobId: string, route: ScheduledRoute, leaseId: string): void {
    const first = route.stages[0];
    if (!first) throw new Error("Cannot assign an empty route");
    this.database.transaction(() => {
      this.database.raw
        .prepare(
          `UPDATE jobs
           SET status = 'leasing', worker_id = ?, deployment_id = ?, lease_id = ?,
               model_digest = ?, updated_at = ?
           WHERE id = ? AND status = 'queued'`,
        )
        .run(
          first.workerId,
          first.deploymentId,
          leaseId,
          first.modelDigest,
          Date.now(),
          jobId,
        );
      this.queueJob(jobId);
    });
  }

  setJobStatus(jobId: string, status: JobStatus, failureCode?: string): void {
    this.database.transaction(() => {
      this.database.raw
        .prepare("UPDATE jobs SET status = ?, failure_code = ?, updated_at = ? WHERE id = ?")
        .run(status, failureCode ?? null, Date.now(), jobId);
      this.queueJob(jobId);
    });
  }

  requeueJob(jobId: string, failureCode?: string): void {
    this.database.transaction(() => {
      this.database.raw
        .prepare(
          `UPDATE jobs
           SET status = 'queued', worker_id = NULL, deployment_id = NULL,
               model_digest = NULL, lease_id = NULL, failure_code = ?, updated_at = ?
           WHERE id = ? AND status NOT IN ('completed', 'cancelled', 'expired')`,
        )
        .run(failureCode ?? null, Date.now(), jobId);
      this.queueJob(jobId);
    });
  }

  completeJob(jobId: string, metrics: CompletionMetrics): void {
    this.database.transaction(() => {
      this.database.raw
        .prepare(
          `UPDATE jobs
           SET status = 'completed', input_tokens = ?, output_tokens = ?, updated_at = ?
           WHERE id = ?`,
        )
        .run(metrics.inputTokens, metrics.outputTokens, Date.now(), jobId);
      this.database.raw
        .prepare(
          `UPDATE workers SET jobs_completed = jobs_completed + 1, updated_at = ?
           WHERE id = (SELECT worker_id FROM jobs WHERE id = ?)`,
        )
        .run(Date.now(), jobId);
      this.queueJob(jobId);
      const workerId = this.getJob(jobId)?.workerId;
      if (workerId) this.queueWorker(workerId);
    });
  }

  countActiveJobs(workerId: string): number {
    const row = this.database.raw
      .prepare(
        `SELECT COUNT(*) AS count FROM jobs
         WHERE worker_id = ? AND status IN ('leasing', 'running', 'streaming')`,
      )
      .get(workerId) as { count: number };
    return Number(row.count);
  }

  listActiveJobsForWorker(workerId: string): StoredJob[] {
    const rows = this.database.raw
      .prepare(
        `SELECT * FROM jobs
         WHERE worker_id = ? AND status IN ('leasing', 'running', 'streaming')`,
      )
      .all(workerId) as unknown as JobRow[];
    return rows.map((row) => this.mapJob(row));
  }

  listNonterminalJobs(): StoredJob[] {
    const rows = this.database.raw
      .prepare(
        `SELECT * FROM jobs
         WHERE status IN ('queued', 'leasing', 'running', 'streaming')`,
      )
      .all() as unknown as JobRow[];
    return rows.map((row) => this.mapJob(row));
  }

  saveSession(sessionId: string, model: string, route: ScheduledRoute, ttlMs = 15 * 60_000): void {
    this.database.transaction(() => {
      const now = Date.now();
      this.database.raw
        .prepare(
          `INSERT INTO sessions(id, model, route_json, expires_at, last_used_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             model = excluded.model,
             route_json = excluded.route_json,
             expires_at = excluded.expires_at,
             last_used_at = excluded.last_used_at,
             version = sessions.version + 1`,
        )
        .run(sessionId, model, JSON.stringify(route), now + ttlMs, now);
      const row = this.database.raw.prepare("SELECT * FROM sessions WHERE id = ?").get(sessionId) as
        | {
            id: string;
            model: string;
            route_json: string;
            expires_at: number;
            last_used_at: number;
            version: number;
          }
        | undefined;
      if (row) {
        this.database.enqueueRemoteChange("sessions", sessionId, "upsert", {
          id: row.id,
          model: row.model,
          route_json: JSON.parse(row.route_json) as unknown,
          expires_at: Number(row.expires_at),
          last_used_at: Number(row.last_used_at),
          version: Number(row.version),
        });
      }
    });
  }

  getSessionRoute(sessionId: string, model: string): ScheduledRoute | null {
    const row = this.database.raw
      .prepare(
        `SELECT route_json FROM sessions
         WHERE id = ? AND model = ? AND expires_at > ?`,
      )
      .get(sessionId, model, Date.now()) as { route_json: string } | undefined;
    return row ? (JSON.parse(row.route_json) as ScheduledRoute) : null;
  }

  listJobs(limit = 100): StoredJob[] {
    const rows = this.database.raw
      .prepare("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?")
      .all(limit) as unknown as JobRow[];
    return rows.map((row) => this.mapJob(row));
  }

  queueAllForRemotePersistence(): number {
    return this.database.transaction(() => {
      let queued = 0;
      const workers = this.database.raw.prepare("SELECT id FROM workers").all() as unknown as Array<{ id: string }>;
      for (const worker of workers) {
        this.queueWorker(worker.id);
        queued += 1;
      }
      const jobs = this.database.raw.prepare("SELECT id FROM jobs").all() as unknown as Array<{ id: string }>;
      for (const job of jobs) {
        this.queueJob(job.id);
        queued += 1;
      }
      const models = this.database.raw.prepare("SELECT id FROM requested_models").all() as unknown as Array<{ id: string }>;
      for (const model of models) {
        this.queueRequestedModel(model.id);
        queued += 1;
      }
      const sessions = this.database.raw.prepare("SELECT * FROM sessions").all() as unknown as Array<{
        id: string;
        model: string;
        route_json: string;
        expires_at: number;
        last_used_at: number;
        version: number;
      }>;
      for (const row of sessions) {
        this.database.enqueueRemoteChange("sessions", row.id, "upsert", {
          id: row.id,
          model: row.model,
          route_json: JSON.parse(row.route_json) as unknown,
          expires_at: Number(row.expires_at),
          last_used_at: Number(row.last_used_at),
          version: Number(row.version),
        });
        queued += 1;
      }
      const keys = this.database.raw.prepare("SELECT * FROM idempotency_keys").all() as unknown as Array<{
        idempotency_key: string;
        request_hash: string;
        job_id: string;
        created_at: number;
      }>;
      for (const row of keys) {
        this.database.enqueueRemoteChange("idempotency_keys", row.idempotency_key, "upsert", {
          ...row,
          created_at: Number(row.created_at),
        });
        queued += 1;
      }
      const conversations = this.database.raw.prepare(
        "SELECT id FROM inference_conversations",
      ).all() as unknown as Array<{ id: string }>;
      for (const row of conversations) {
        this.queueInferenceConversation(row.id);
        queued += 1;
      }
      const messages = this.database.raw.prepare(
        "SELECT id FROM inference_messages",
      ).all() as unknown as Array<{ id: string }>;
      for (const row of messages) {
        this.queueInferenceMessage(row.id);
        queued += 1;
      }
      const activationEvents = this.database.raw.prepare(
        "SELECT id FROM activation_events",
      ).all() as unknown as Array<{ id: string }>;
      for (const row of activationEvents) {
        this.queueActivationEvent(row.id);
        queued += 1;
      }
      const diagnosticEvents = this.database.raw.prepare(
        "SELECT id FROM diagnostic_events",
      ).all() as unknown as Array<{ id: string }>;
      for (const row of diagnosticEvents) {
        this.queueDiagnosticEvent(row.id);
        queued += 1;
      }
      const deploymentStates = this.database.raw.prepare(
        "SELECT * FROM deployment_states",
      ).all() as unknown as Array<{
        model_id: string;
        desired_state: string;
        observed_state: string;
        generation: number;
        observed_generation: number;
        retry_count: number;
        next_retry_at: number | null;
        last_error: string | null;
        active_operation_id: string | null;
        controller_owner: string | null;
        controller_lease_until: number | null;
        created_at: number;
        updated_at: number;
      }>;
      for (const row of deploymentStates) {
        this.database.enqueueRemoteChange("deployment_states", row.model_id, "upsert", row);
        queued += 1;
      }
      const deploymentOperations = this.database.raw.prepare(
        "SELECT * FROM deployment_operations",
      ).all() as unknown as Array<{
        id: string;
        model_id: string;
        generation: number;
        kind: string;
        status: string;
        attempt: number;
        idempotency_key: string;
        error_code: string | null;
        error_message: string | null;
        metadata_json: string;
        started_at: number;
        updated_at: number;
        finished_at: number | null;
      }>;
      for (const row of deploymentOperations) {
        const { metadata_json: metadataJson, ...operation } = row;
        this.database.enqueueRemoteChange("deployment_operations", row.id, "upsert", {
          ...operation,
          metadata: JSON.parse(metadataJson) as unknown,
        });
        queued += 1;
      }
      const routeReservations = this.database.raw.prepare(
        "SELECT * FROM route_reservations",
      ).all() as unknown as Array<{
        id: string;
        model_id: string;
        operation_id: string;
        generation: number;
        status: string;
        route_digest: string;
        stages_json: string;
        canary_json: string | null;
        expires_at: number;
        committed_at: number | null;
        released_at: number | null;
        error: string | null;
        created_at: number;
        updated_at: number;
      }>;
      for (const row of routeReservations) {
        const {
          stages_json: stagesJson,
          canary_json: canaryJson,
          ...reservation
        } = row;
        this.database.enqueueRemoteChange("route_reservations", row.id, "upsert", {
          ...reservation,
          stages: JSON.parse(stagesJson) as unknown,
          canary: canaryJson ? JSON.parse(canaryJson) as unknown : null,
        });
        queued += 1;
      }
      const deploymentStageLeases = this.database.raw.prepare(
        "SELECT * FROM deployment_stage_leases",
      ).all() as unknown as Array<Record<string, unknown> & { id: string }>;
      for (const row of deploymentStageLeases) {
        this.database.enqueueRemoteChange("deployment_stage_leases", row.id, "upsert", row);
        queued += 1;
      }
      if (this.database.raw.prepare(
        "SELECT 1 FROM federation_settings WHERE id = 'global'",
      ).get()) {
        this.queueFederationSettings();
        queued += 1;
      }
      const networkSettings = this.database.raw.prepare(
        "SELECT id FROM federated_network_settings",
      ).all() as unknown as Array<{ id: FederatedNetworkId }>;
      for (const row of networkSettings) {
        this.queueFederatedNetworkSettings(row.id);
        queued += 1;
      }
      for (const table of [
        "federated_route_attempts",
        "provider_spend_reservations",
        "managed_rentals",
      ] as const) {
        const rows = this.database.raw.prepare(
          `SELECT id FROM ${table}`,
        ).all() as unknown as Array<{ id: string }>;
        for (const row of rows) {
          this.queueFederationTableRow(table, row.id);
          queued += 1;
        }
      }
      return queued;
    });
  }

  queueBenchmarkRun(run: BenchmarkRun): void {
    this.database.enqueueRemoteChange("benchmark_runs", run.runId, "upsert", {
      run_id: run.runId,
      version: run.version,
      label: run.label,
      status: run.status,
      trigger: run.trigger ?? null,
      trigger_model_id: run.triggerModelId ?? null,
      started_at: run.startedAt,
      finished_at: run.finishedAt,
      document: run,
    });
  }

  startInferenceConversation(
    sessionId: string,
    model: string,
    messages: ReadonlyArray<{ role: "system" | "developer" | "user" | "assistant" | "tool"; content: string }>,
  ): string {
    return this.database.transaction(() => {
      const now = Date.now();
      const existing = this.database.raw.prepare(
        "SELECT id FROM inference_conversations WHERE session_id = ?",
      ).get(sessionId) as { id: string } | undefined;
      const conversationId = existing?.id ?? newId("cnv");
      if (existing) {
        this.database.raw.prepare(
          "UPDATE inference_conversations SET model = ?, updated_at = ? WHERE id = ?",
        ).run(model, now, conversationId);
      } else {
        this.database.raw.prepare(
          `INSERT INTO inference_conversations(id, session_id, model, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)`,
        ).run(conversationId, sessionId, model, now, now);
      }
      this.queueInferenceConversation(conversationId);

      const messageCount = this.database.raw.prepare(
        "SELECT COUNT(*) AS count FROM inference_messages WHERE conversation_id = ?",
      ).get(conversationId) as { count: number };
      const selected = Number(messageCount.count) === 0
        ? messages
        : messages.length > 0 ? [messages.at(-1)!] : [];
      for (const message of selected) {
        this.appendInferenceMessage({
          conversationId,
          jobId: null,
          role: message.role,
          content: message.content,
          status: "completed",
        });
      }
      return conversationId;
    });
  }

  appendInferenceMessage(input: {
    conversationId: string;
    jobId: string | null;
    role: StoredInferenceMessage["role"];
    content: string;
    status: StoredInferenceMessage["status"];
    inputTokens?: number | null;
    outputTokens?: number | null;
    routeClass?: string | null;
    latencyMs?: number | null;
    metadata?: Record<string, unknown>;
  }): StoredInferenceMessage {
    return this.database.transaction(() => {
      const id = newId("msg");
      const createdAt = Date.now();
      this.database.raw.prepare(
        `INSERT INTO inference_messages(
           id, conversation_id, job_id, role, content, status, input_tokens,
           output_tokens, route_class, latency_ms, metadata_json, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        id,
        input.conversationId,
        input.jobId,
        input.role,
        input.content,
        input.status,
        input.inputTokens ?? null,
        input.outputTokens ?? null,
        input.routeClass ?? null,
        input.latencyMs ?? null,
        JSON.stringify(input.metadata ?? {}),
        createdAt,
      );
      this.database.raw.prepare(
        "UPDATE inference_conversations SET updated_at = ? WHERE id = ?",
      ).run(createdAt, input.conversationId);
      this.queueInferenceConversation(input.conversationId);
      this.queueInferenceMessage(id);
      return {
        id,
        conversationId: input.conversationId,
        jobId: input.jobId,
        role: input.role,
        content: input.content,
        status: input.status,
        inputTokens: input.inputTokens ?? null,
        outputTokens: input.outputTokens ?? null,
        routeClass: input.routeClass ?? null,
        latencyMs: input.latencyMs ?? null,
        metadata: input.metadata ?? {},
        createdAt,
      };
    });
  }

  listInferenceMessages(sessionId: string): StoredInferenceMessage[] {
    const rows = this.database.raw.prepare(
      `SELECT m.*
       FROM inference_messages m
       JOIN inference_conversations c ON c.id = m.conversation_id
       WHERE c.session_id = ?
       ORDER BY m.created_at, m.id`,
    ).all(sessionId) as unknown as Array<{
      id: string;
      conversation_id: string;
      job_id: string | null;
      role: StoredInferenceMessage["role"];
      content: string;
      status: StoredInferenceMessage["status"];
      input_tokens: number | null;
      output_tokens: number | null;
      route_class: string | null;
      latency_ms: number | null;
      metadata_json: string;
      created_at: number;
    }>;
    return rows.map((row) => ({
      id: row.id,
      conversationId: row.conversation_id,
      jobId: row.job_id,
      role: row.role,
      content: row.content,
      status: row.status,
      inputTokens: row.input_tokens === null ? null : Number(row.input_tokens),
      outputTokens: row.output_tokens === null ? null : Number(row.output_tokens),
      routeClass: row.route_class,
      latencyMs: row.latency_ms === null ? null : Number(row.latency_ms),
      metadata: JSON.parse(row.metadata_json) as Record<string, unknown>,
      createdAt: Number(row.created_at),
    }));
  }

  appendActivationEvent(modelId: string, event: StoredActivationEvent): void {
    this.database.transaction(() => {
      const id = newId("act");
      const occurredAt = Date.parse(event.at);
      this.database.raw.prepare(
        `INSERT INTO activation_events(
           id, model_id, phase, state, message, node_id, process_id, device,
           details_json, occurred_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        id,
        modelId,
        event.phase,
        event.state,
        event.message,
        event.nodeId ?? null,
        event.processId ?? null,
        event.device ?? null,
        event.details ? JSON.stringify(event.details) : null,
        Number.isFinite(occurredAt) ? occurredAt : Date.now(),
      );
      this.queueActivationEvent(id);
    });
  }

  listActivationEvents(modelId: string, limit = 100): StoredActivationEvent[] {
    const rows = this.database.raw.prepare(
      `SELECT * FROM activation_events
       WHERE model_id = ?
       ORDER BY occurred_at DESC
       LIMIT ?`,
    ).all(modelId, limit) as unknown as Array<{
      phase: string;
      state: StoredActivationEvent["state"];
      message: string;
      node_id: string | null;
      process_id: string | null;
      device: string | null;
      details_json: string | null;
      occurred_at: number;
    }>;
    return rows.toReversed().map((row) => ({
      phase: row.phase,
      state: row.state,
      message: row.message,
      at: new Date(Number(row.occurred_at)).toISOString(),
      ...(row.node_id ? { nodeId: row.node_id } : {}),
      ...(row.process_id ? { processId: row.process_id } : {}),
      ...(row.device ? { device: row.device } : {}),
      ...(row.details_json ? { details: JSON.parse(row.details_json) as string[] } : {}),
    }));
  }

  appendDiagnosticEvents(
    events: readonly RemoteDiagnosticEvent[],
  ): { accepted: number; duplicates: number } {
    return this.database.transaction(() => {
      const insert = this.database.raw.prepare(
        `INSERT OR IGNORE INTO diagnostic_events(
           id, source_id, app_version, platform, arch, level, source, event,
           message, details, occurred_at, received_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      );
      const receivedAt = Date.now();
      let accepted = 0;
      for (const event of events) {
        const occurredAt = Date.parse(event.occurredAt);
        const safeOccurredAt = Number.isFinite(occurredAt)
          ? Math.min(occurredAt, receivedAt + 5 * 60_000)
          : receivedAt;
        const result = insert.run(
          event.id,
          event.sourceId,
          event.appVersion,
          event.platform,
          event.arch,
          event.level,
          event.source,
          event.event,
          event.message,
          event.details ?? null,
          safeOccurredAt,
          receivedAt,
        );
        if (Number(result.changes) === 0) continue;
        accepted += 1;
        this.queueDiagnosticEvent(event.id);
      }
      this.pruneDiagnosticEvents(receivedAt);
      return { accepted, duplicates: events.length - accepted };
    });
  }

  listDiagnosticEvents(query: DiagnosticEventQuery = {}): StoredDiagnosticEvent[] {
    const where: string[] = [];
    const parameters: Array<string | number> = [];
    if (query.level) {
      where.push("level = ?");
      parameters.push(query.level);
    }
    if (query.sourceId) {
      where.push("source_id = ?");
      parameters.push(query.sourceId);
    }
    if (query.since) {
      const since = Date.parse(query.since);
      if (Number.isFinite(since)) {
        where.push("occurred_at >= ?");
        parameters.push(since);
      }
    }
    const limit = Math.max(1, Math.min(500, Math.trunc(query.limit ?? 100)));
    const rows = this.database.raw.prepare(
      `SELECT * FROM diagnostic_events
       ${where.length > 0 ? `WHERE ${where.join(" AND ")}` : ""}
       ORDER BY occurred_at DESC, id DESC
       LIMIT ?`,
    ).all(...parameters, limit) as unknown as Array<{
      id: string;
      source_id: string;
      app_version: string;
      platform: string;
      arch: string;
      level: RemoteDiagnosticEvent["level"];
      source: RemoteDiagnosticEvent["source"];
      event: string;
      message: string;
      details: string | null;
      occurred_at: number;
      received_at: number;
    }>;
    return rows.map((row) => ({
      id: row.id,
      sourceId: row.source_id,
      appVersion: row.app_version,
      platform: row.platform,
      arch: row.arch,
      level: row.level,
      source: row.source,
      event: row.event,
      message: row.message,
      ...(row.details ? { details: row.details } : {}),
      occurredAt: new Date(Number(row.occurred_at)).toISOString(),
      receivedAt: new Date(Number(row.received_at)).toISOString(),
    }));
  }

  recordNetworkTelemetrySample(sample: StoredNetworkTelemetrySample): void {
    this.database.raw.prepare(
      `INSERT INTO network_telemetry_history(
         captured_at, registered_nodes, connected_nodes, online_nodes, browser_nodes,
         active_models, model_replicas, model_pipelines, offered_vram_mb, free_vram_mb,
         inflight_jobs, running_jobs, completed_jobs
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(captured_at) DO UPDATE SET
         registered_nodes=excluded.registered_nodes,
         connected_nodes=excluded.connected_nodes,
         online_nodes=excluded.online_nodes,
         browser_nodes=excluded.browser_nodes,
         active_models=excluded.active_models,
         model_replicas=excluded.model_replicas,
         model_pipelines=excluded.model_pipelines,
         offered_vram_mb=excluded.offered_vram_mb,
         free_vram_mb=excluded.free_vram_mb,
         inflight_jobs=excluded.inflight_jobs,
         running_jobs=excluded.running_jobs,
         completed_jobs=excluded.completed_jobs`,
    ).run(
      sample.capturedAt,
      sample.registeredNodes,
      sample.connectedNodes,
      sample.onlineNodes,
      sample.browserNodes,
      sample.activeModels,
      sample.modelReplicas,
      sample.modelPipelines,
      sample.offeredVramMb,
      sample.freeVramMb,
      sample.inflightJobs,
      sample.runningJobs,
      sample.completedJobs,
    );
  }

  listNetworkTelemetrySamples(since: number, limit = 13_000): StoredNetworkTelemetrySample[] {
    const boundedLimit = Math.max(1, Math.min(13_000, Math.trunc(limit)));
    const rows = this.database.raw.prepare(
      `SELECT * FROM (
         SELECT captured_at, registered_nodes, connected_nodes, online_nodes, browser_nodes,
                active_models, model_replicas, model_pipelines, offered_vram_mb, free_vram_mb,
                inflight_jobs, running_jobs, completed_jobs
         FROM network_telemetry_history
         WHERE captured_at >= ?
         ORDER BY captured_at DESC
         LIMIT ?
       ) ORDER BY captured_at ASC`,
    ).all(since, boundedLimit) as unknown as Array<{
      captured_at: number;
      registered_nodes: number;
      connected_nodes: number;
      online_nodes: number;
      browser_nodes: number;
      active_models: number;
      model_replicas: number;
      model_pipelines: number;
      offered_vram_mb: number;
      free_vram_mb: number;
      inflight_jobs: number;
      running_jobs: number;
      completed_jobs: number;
    }>;
    return rows.map((row) => ({
      capturedAt: Number(row.captured_at),
      registeredNodes: Number(row.registered_nodes),
      connectedNodes: Number(row.connected_nodes),
      onlineNodes: Number(row.online_nodes),
      browserNodes: Number(row.browser_nodes),
      activeModels: Number(row.active_models),
      modelReplicas: Number(row.model_replicas),
      modelPipelines: Number(row.model_pipelines),
      offeredVramMb: Number(row.offered_vram_mb),
      freeVramMb: Number(row.free_vram_mb),
      inflightJobs: Number(row.inflight_jobs),
      runningJobs: Number(row.running_jobs),
      completedJobs: Number(row.completed_jobs),
    }));
  }

  pruneNetworkTelemetrySamples(before: number): number {
    return Number(
      this.database.raw.prepare(
        "DELETE FROM network_telemetry_history WHERE captured_at < ?",
      ).run(before).changes,
    );
  }

  private queueFederationSettings(): void {
    const row = this.database.raw.prepare(
      "SELECT * FROM federation_settings WHERE id = 'global'",
    ).get() as Record<string, unknown> | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("federation_settings", "global", "upsert", {
      ...row,
      enabled: Number(row.enabled) === 1,
      autoscaling_enabled: Number(row.autoscaling_enabled) === 1,
    });
  }

  private queueFederatedNetworkSettings(id: string): void {
    const row = this.database.raw.prepare(
      "SELECT * FROM federated_network_settings WHERE id = ?",
    ).get(id) as Record<string, unknown> | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("federated_network_settings", id, "upsert", {
      ...row,
      enabled: Number(row.enabled) === 1,
    });
  }

  private queueFederationTableRow(
    table:
      | "federated_route_attempts"
      | "provider_spend_reservations"
      | "managed_rentals",
    id: string,
  ): void {
    const row = this.database.raw.prepare(
      `SELECT * FROM ${table} WHERE id = ?`,
    ).get(id) as Record<string, unknown> | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange(table, id, "upsert", table === "managed_rentals"
      ? {
          ...row,
          hardware_json: JSON.parse(String(row.hardware_json)),
          labels_json: JSON.parse(String(row.labels_json)),
        }
      : row);
  }

  private queueWorker(workerId: string): void {
    const row = this.database.raw.prepare("SELECT * FROM workers WHERE id = ?").get(workerId) as
      | (WorkerRow & {
          created_at: number;
          updated_at: number;
          deregistered: number;
        })
      | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("workers", workerId, "upsert", {
      id: row.id,
      status: row.status,
      capabilities_json: JSON.parse(row.capabilities_json) as unknown,
      reliability: Number(row.reliability),
      jobs_completed: Number(row.jobs_completed),
      last_seen_at: Number(row.last_seen_at),
      created_at: Number(row.created_at),
      updated_at: Number(row.updated_at),
      deregistered: row.deregistered === 1,
      identity_kind: row.identity_kind,
      identity_id: row.identity_id,
    });
  }

  private queueJob(jobId: string): void {
    const row = this.database.raw.prepare("SELECT * FROM jobs WHERE id = ?").get(jobId) as
      | JobRow
      | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("jobs", jobId, "upsert", {
      id: row.id,
      session_id: row.session_id,
      model: row.model,
      workload_class: row.workload_class,
      status: row.status,
      worker_id: row.worker_id,
      deployment_id: row.deployment_id,
      model_digest: row.model_digest,
      lease_id: row.lease_id,
      input_tokens: Number(row.input_tokens),
      output_tokens: Number(row.output_tokens),
      failure_code: row.failure_code,
      deadline_at: Number(row.deadline_at),
      created_at: Number(row.created_at),
      updated_at: Number(row.updated_at),
    });
  }

  private queueRequestedModel(modelId: string): void {
    const row = this.database.raw.prepare("SELECT * FROM requested_models WHERE id = ?").get(modelId) as
      | RequestedModelRow
      | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("requested_models", modelId, "upsert", {
      id: row.id,
      source: row.source,
      revision: row.revision,
      context_tokens: Number(row.context_tokens),
      minimum_nodes: Number(row.minimum_nodes),
      auto_activate: row.auto_activate === 1,
      profile_json: row.profile_json ? JSON.parse(row.profile_json) as unknown : null,
      profile_error: row.profile_error,
      activation_requested_at: row.activation_requested_at,
      activation_error: row.activation_error,
      created_at: Number(row.created_at),
      updated_at: Number(row.updated_at),
    });
  }

  private queueInferenceConversation(conversationId: string): void {
    const row = this.database.raw.prepare(
      "SELECT * FROM inference_conversations WHERE id = ?",
    ).get(conversationId) as
      | { id: string; session_id: string; model: string; created_at: number; updated_at: number }
      | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("inference_conversations", conversationId, "upsert", {
      id: row.id,
      session_id: row.session_id,
      model: row.model,
      created_at: new Date(Number(row.created_at)).toISOString(),
      updated_at: new Date(Number(row.updated_at)).toISOString(),
    });
  }

  private queueInferenceMessage(messageId: string): void {
    const row = this.database.raw.prepare(
      "SELECT * FROM inference_messages WHERE id = ?",
    ).get(messageId) as
      | {
          id: string;
          conversation_id: string;
          job_id: string | null;
          role: string;
          content: string;
          status: string;
          input_tokens: number | null;
          output_tokens: number | null;
          route_class: string | null;
          latency_ms: number | null;
          metadata_json: string;
          created_at: number;
        }
      | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("inference_messages", messageId, "upsert", {
      id: row.id,
      conversation_id: row.conversation_id,
      job_id: row.job_id,
      role: row.role,
      content: row.content,
      status: row.status,
      input_tokens: row.input_tokens,
      output_tokens: row.output_tokens,
      route_class: row.route_class,
      latency_ms: row.latency_ms,
      metadata: JSON.parse(row.metadata_json) as unknown,
      created_at: new Date(Number(row.created_at)).toISOString(),
    });
  }

  private queueActivationEvent(eventId: string): void {
    const row = this.database.raw.prepare(
      "SELECT * FROM activation_events WHERE id = ?",
    ).get(eventId) as
      | {
          id: string;
          model_id: string;
          phase: string;
          state: string;
          message: string;
          node_id: string | null;
          process_id: string | null;
          device: string | null;
          details_json: string | null;
          occurred_at: number;
        }
      | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("activation_events", eventId, "upsert", {
      id: row.id,
      model_id: row.model_id,
      phase: row.phase,
      state: row.state,
      message: row.message,
      node_id: row.node_id,
      process_id: row.process_id,
      device: row.device,
      details: row.details_json ? JSON.parse(row.details_json) as unknown : null,
      occurred_at: Number(row.occurred_at),
    });
  }

  private queueDiagnosticEvent(eventId: string): void {
    const row = this.database.raw.prepare(
      "SELECT * FROM diagnostic_events WHERE id = ?",
    ).get(eventId) as {
      id: string;
      source_id: string;
      app_version: string;
      platform: string;
      arch: string;
      level: string;
      source: string;
      event: string;
      message: string;
      details: string | null;
      occurred_at: number;
      received_at: number;
    } | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("diagnostic_events", eventId, "upsert", row);
  }

  private pruneDiagnosticEvents(now: number): void {
    const cutoff = now - 30 * 24 * 60 * 60 * 1_000;
    const expired = this.database.raw.prepare(
      "SELECT id FROM diagnostic_events WHERE occurred_at < ?",
    ).all(cutoff) as unknown as Array<{ id: string }>;
    const overflow = this.database.raw.prepare(
      `SELECT id FROM diagnostic_events
       ORDER BY occurred_at DESC, id DESC
       LIMIT -1 OFFSET 10000`,
    ).all() as unknown as Array<{ id: string }>;
    const ids = new Set([...expired, ...overflow].map((row) => row.id));
    const remove = this.database.raw.prepare(
      "DELETE FROM diagnostic_events WHERE id = ?",
    );
    for (const id of ids) {
      remove.run(id);
      this.database.enqueueRemoteChange("diagnostic_events", id, "delete", null);
    }
  }

  private mapWorker(row: WorkerRow): StoredWorker {
    return {
      id: row.id,
      status: row.status,
      capabilities: JSON.parse(row.capabilities_json) as WorkerCapabilities,
      reliability: Number(row.reliability),
      jobsCompleted: Number(row.jobs_completed),
      lastSeenAt: Number(row.last_seen_at),
      identityKind: row.identity_kind,
      identityId: row.identity_id,
    };
  }

  private mapRequestedModel(row: RequestedModelRow): StoredRequestedModel {
    return {
      id: row.id,
      source: row.source,
      revision: row.revision,
      contextTokens: row.context_tokens,
      minimumNodes: row.minimum_nodes,
      autoActivate: row.auto_activate === 1,
      profile: row.profile_json ? JSON.parse(row.profile_json) as Record<string, unknown> : null,
      profileError: row.profile_error,
      activationRequestedAt: row.activation_requested_at,
      activationError: row.activation_error,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  }

  private mapJob(row: JobRow): StoredJob {
    return {
      id: row.id,
      sessionId: row.session_id,
      model: row.model,
      workloadClass: row.workload_class,
      status: row.status,
      workerId: row.worker_id,
      deploymentId: row.deployment_id,
      modelDigest: row.model_digest,
      leaseId: row.lease_id,
      inputTokens: Number(row.input_tokens),
      outputTokens: Number(row.output_tokens),
      failureCode: row.failure_code,
      deadlineAt: Number(row.deadline_at),
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    };
  }
}
