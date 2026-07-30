import { createHash } from "node:crypto";
import type {
  FederatedNetworkId,
  ManagedRental,
} from "../contracts/federation.js";
import type { FederationRuntimeConfig } from "../core/config.js";
import { newId } from "../core/ids.js";
import type { MeshStore } from "../storage/store.js";
import type { WorkerHub } from "./worker-hub.js";

const RENTAL_PROVIDERS = ["gpu_cloud", "vast", "clore"] as const;
type RentalProviderId = typeof RENTAL_PROVIDERS[number];

export interface RentalQuote {
  offerId: string;
  hourlyUsd: number;
  gpuModel: string;
  vramMb: number;
  reliability: number;
  verifiedHost: boolean;
  spot: boolean;
}

export interface RentalProviderDriver {
  readonly id: RentalProviderId;
  readonly configured: boolean;
  quote(input: {
    minimumVramMb: number;
    spot: boolean;
    signal: AbortSignal;
  }): Promise<RentalQuote | null>;
  create(input: {
    quote: RentalQuote;
    name: string;
    image: string;
    labels: Record<string, string>;
    environment: Record<string, string>;
    signal: AbortSignal;
  }): Promise<{ externalId: string }>;
  terminate(externalId: string, signal: AbortSignal): Promise<void>;
  exists(externalId: string, signal: AbortSignal): Promise<boolean>;
}

export interface RentalCapacityManagerOptions {
  now?: () => number;
  reconcileIntervalMs?: number;
}

interface PendingDemand {
  model: string;
  minimumVramMb: number;
  batch: boolean;
  projectedTokenCostUsdPerHour: number;
  firstSeenAt: number;
  lastSeenAt: number;
}

export class RentalCapacityManager {
  private readonly drivers = new Map<RentalProviderId, RentalProviderDriver>();
  private readonly now: () => number;
  private readonly demands = new Map<string, PendingDemand>();
  private timer: NodeJS.Timeout | null = null;
  private lastScaleAt = 0;
  private reconciling = false;

  constructor(
    private readonly store: MeshStore,
    private readonly hub: WorkerHub,
    drivers: readonly RentalProviderDriver[],
    private readonly coordinatorUrl: string | undefined,
    private readonly image: string | undefined,
    options: RentalCapacityManagerOptions = {},
  ) {
    for (const driver of drivers) this.drivers.set(driver.id, driver);
    this.now = options.now ?? Date.now;
    this.reconcileIntervalMs = options.reconcileIntervalMs ?? 60_000;
  }

  private readonly reconcileIntervalMs: number;

  start(): void {
    if (this.timer) return;
    void this.reconcile();
    this.timer = setInterval(() => void this.reconcile(), this.reconcileIntervalMs);
    this.timer.unref();
  }

  configuredProviders(): Record<RentalProviderId, boolean> {
    return Object.fromEntries(RENTAL_PROVIDERS.map((id) => [
      id,
      Boolean(this.drivers.get(id)?.configured && this.image && this.coordinatorUrl),
    ])) as Record<RentalProviderId, boolean>;
  }

  recordUnmetDemand(input: {
    model: string;
    minimumVramMb: number;
    batch: boolean;
    projectedTokenCostUsdPerHour: number;
  }): void {
    const now = this.now();
    const existing = this.demands.get(input.model);
    this.demands.set(input.model, {
      ...input,
      firstSeenAt: existing?.firstSeenAt ?? now,
      lastSeenAt: now,
    });
  }

  async reconcile(): Promise<void> {
    if (this.reconciling) return;
    this.reconciling = true;
    try {
      await this.reconcileRentals();
      await this.maybeScale();
    } finally {
      this.reconciling = false;
    }
  }

  async emergencyStop(): Promise<void> {
    await Promise.allSettled(
      this.store.listManagedRentals()
        .filter((rental) => !["stopped", "failed"].includes(rental.state))
        .map((rental) => this.stopRental(rental.id, "emergency_stop")),
    );
  }

  close(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  private async maybeScale(): Promise<void> {
    const settings = this.store.getFederationSettings(false);
    if (
      !settings.enabled
      || !settings.autoscalingEnabled
      || settings.dailyBudgetUsd <= 0
      || settings.monthlyBudgetUsd <= 0
      || !this.coordinatorUrl
      || !this.image
      || !this.image.includes("@sha256:")
      || this.now() - this.lastScaleAt < 10 * 60_000
    ) return;
    const active = this.store.listManagedRentals().filter(
      (rental) => !["stopped", "failed"].includes(rental.state),
    );
    if (active.length >= settings.maxRentals) return;
    const demand = [...this.demands.values()]
      .filter((entry) =>
        this.now() - entry.firstSeenAt >= 5 * 60_000
        && this.now() - entry.lastSeenAt <= 2 * 60_000)
      .sort((left, right) => right.minimumVramMb - left.minimumVramMb)[0];
    if (!demand) return;
    for (const provider of RENTAL_PROVIDERS) {
      const driver = this.drivers.get(provider);
      if (!driver?.configured) continue;
      const network = this.store.getFederatedNetworkSettings(provider, {
        enabled: false,
        priority: 500,
        dailyBudgetUsd: 0,
        monthlyBudgetUsd: 0,
      });
      if (!network.enabled || network.dailyBudgetUsd <= 0 || network.monthlyBudgetUsd <= 0) {
        continue;
      }
      const controller = timeoutController(20_000);
      try {
        const quote = await driver.quote({
          minimumVramMb: demand.minimumVramMb,
          spot: demand.batch,
          signal: controller.signal,
        });
        if (
          !quote
          || !quote.verifiedHost
          || quote.reliability < 0.98
          || quote.vramMb < demand.minimumVramMb
          || quote.spot && !demand.batch
          || demand.projectedTokenCostUsdPerHour < quote.hourlyUsd * 1.25
        ) continue;
        await this.createRental(driver, quote, demand);
        this.lastScaleAt = this.now();
        return;
      } finally {
        controller.abort();
      }
    }
  }

  private async createRental(
    driver: RentalProviderDriver,
    quote: RentalQuote,
    demand: PendingDemand,
  ): Promise<void> {
    if (!this.image || !this.coordinatorUrl) return;
    const internalId = newId("rental");
    const publicId = `rent_${createHash("sha256").update(internalId).digest("hex").slice(0, 20)}`;
    const nodeId = `rental-${publicId.slice(-20)}`;
    const reservationId = newId("rent-spend");
    const settings = this.store.getFederationSettings(false);
    const network = this.store.getFederatedNetworkSettings(driver.id, {
      enabled: false,
      priority: 500,
      dailyBudgetUsd: 0,
      monthlyBudgetUsd: 0,
    });
    const maximumUsd = quote.hourlyUsd;
    if (!this.store.reserveProviderSpend({
      id: reservationId,
      provider: driver.id,
      requestId: publicId,
      maximumUsd,
      dailyBudgetUsd: Math.min(settings.dailyBudgetUsd, network.dailyBudgetUsd),
      monthlyBudgetUsd: Math.min(settings.monthlyBudgetUsd, network.monthlyBudgetUsd),
    })) return;
    const labels = {
      "mycellios-managed": "true",
      "mycellios-rental-id": publicId,
    };
    const now = this.now();
    try {
      const created = await driver.create({
        quote,
        name: `mycellios-${publicId}`,
        image: this.image,
        labels,
        environment: {
          GPU_MESH_COORDINATOR: this.coordinatorUrl,
          MYCELLIOS_NODE_ID: nodeId,
          MYCELLIOS_WORKER_CREDENTIAL_PATH: "/var/lib/mycellios/worker-credential.json",
          MYCELLIOS_RENTAL_ID: publicId,
          MYCELLIOS_RENTAL_MODEL: demand.model,
        },
        signal: timeoutController(60_000).signal,
      });
      this.store.saveManagedRental({
        id: publicId,
        provider: driver.id,
        state: "awaiting-worker",
        image: this.image,
        requestedHardware: {
          gpuModel: quote.gpuModel,
          minimumVramMb: demand.minimumVramMb,
          spot: quote.spot,
        },
        workerId: null,
        reservedCostUsd: maximumUsd,
        createdAt: now,
        updatedAt: now,
        drainStartedAt: null,
        stoppedAt: null,
        lastError: null,
        externalId: created.externalId,
        credentialIdentityId: null,
        labels,
      });
    } catch (error) {
      this.store.releaseProviderSpend(reservationId);
      throw error;
    }
  }

  private async reconcileRentals(): Promise<void> {
    const now = this.now();
    for (const rental of this.store.listManagedRentals()) {
      if (["stopped", "failed"].includes(rental.state)) continue;
      const privateRental = this.store.getManagedRentalPrivate(rental.id);
      const driver = this.drivers.get(rental.provider);
      if (!privateRental || !driver?.configured) continue;
      if (
        privateRental.labels["mycellios-managed"] !== "true"
        || privateRental.labels["mycellios-rental-id"] !== rental.id
      ) continue;
      const reservedHours = Math.max(
        1,
        Math.ceil((now - rental.createdAt) / (60 * 60_000) + 0.25),
      );
      const requiredReservation = reservedHours * rental.reservedCostUsd;
      const currentReservation = this.store.providerReservedForRequest(rental.id);
      if (currentReservation + 1e-9 < requiredReservation) {
        const settings = this.store.getFederationSettings(false);
        const network = this.store.getFederatedNetworkSettings(rental.provider, {
          enabled: false,
          priority: 500,
          dailyBudgetUsd: 0,
          monthlyBudgetUsd: 0,
        });
        const extended = this.store.reserveProviderSpend({
          id: newId("rent-spend"),
          provider: rental.provider,
          requestId: rental.id,
          maximumUsd: requiredReservation - currentReservation,
          dailyBudgetUsd: Math.min(settings.dailyBudgetUsd, network.dailyBudgetUsd),
          monthlyBudgetUsd: Math.min(settings.monthlyBudgetUsd, network.monthlyBudgetUsd),
        });
        if (!extended) {
          await this.stopRental(rental.id, "budget_exhausted");
          continue;
        }
      }
      const expectedNodeId = `rental-${rental.id.slice(-20)}`;
      const worker = this.store.listWorkers().find((candidate) =>
        candidate.capabilities.distributedExecutor?.nodeId === expectedNodeId
        && candidate.capabilities.distributedExecutor.physicalIdentity
        && candidate.capabilities.deployments.some(
          (deployment) => deployment.verificationState === "verified",
        )
        && this.hub.isConnected(candidate.id));
      if (worker && rental.state === "awaiting-worker") {
        this.store.saveManagedRental({
          ...privateRental,
          state: "verified",
          workerId: worker.id,
          credentialIdentityId: worker.identityId,
          updatedAt: now,
        });
        continue;
      }
      if (!worker && now - rental.createdAt >= 20 * 60_000) {
        await this.stopRental(rental.id, "worker_verification_timeout");
        continue;
      }
      const demandActive = [...this.demands.values()].some(
        (demand) => now - demand.lastSeenAt < 15 * 60_000,
      );
      if (rental.state === "verified" && !demandActive) {
        await this.stopRental(rental.id, "idle_timeout");
        continue;
      }
      const exists = await driver.exists(privateRental.externalId, timeoutController(15_000).signal);
      if (!exists) {
        this.store.saveManagedRental({
          ...privateRental,
          state: "stopped",
          updatedAt: now,
          stoppedAt: now,
          lastError: null,
        });
      }
    }
  }

  private async stopRental(id: string, reason: string): Promise<void> {
    const rental = this.store.getManagedRentalPrivate(id);
    if (!rental || ["stopped", "failed"].includes(rental.state)) return;
    const driver = this.drivers.get(rental.provider);
    if (!driver?.configured) return;
    const now = this.now();
    this.store.saveManagedRental({
      ...rental,
      state: "stopping",
      updatedAt: now,
      drainStartedAt: rental.drainStartedAt ?? now,
      lastError: reason,
    });
    if (rental.workerId) this.hub.removeWorker(rental.workerId);
    const credential = rental.credentialIdentityId
      ? this.store.database.getWorkerAdmissionCredential("device", rental.credentialIdentityId)
      : null;
    if (credential) {
      this.store.database.revokeWorkerAdmissionCredential({
        identityKind: "device",
        identityId: credential.identityId,
        expectedFingerprint: credential.fingerprint,
        reason: `managed_rental_${reason}`,
      });
    }
    const controller = timeoutController(60_000);
    await driver.terminate(rental.externalId, controller.signal);
    const stillExists = await driver.exists(rental.externalId, controller.signal);
    if (stillExists) throw new Error(`${rental.provider}_teardown_not_confirmed`);
    const actualUsd = Math.max(
      0,
      (this.now() - rental.createdAt) / (60 * 60_000) * rental.reservedCostUsd,
    );
    this.store.reconcileProviderSpendForRequest(rental.id, actualUsd);
    this.store.saveManagedRental({
      ...rental,
      state: "stopped",
      updatedAt: this.now(),
      drainStartedAt: rental.drainStartedAt ?? now,
      stoppedAt: this.now(),
      lastError: reason,
    });
  }
}

export function createRentalProviderDrivers(
  config: FederationRuntimeConfig,
): RentalProviderDriver[] {
  return [
    new GpuCloudRentalDriver(
      config.gpu_cloudApiKey,
      config.gpu_cloudOrganizationId,
      config.gpu_cloudProjectId,
      config.gpu_cloudMaximumHourlyUsd,
    ),
    new VastRentalDriver(config.vastApiKey),
    new CloreRentalDriver(config.cloreApiKey),
  ];
}

abstract class JsonRentalDriver implements RentalProviderDriver {
  abstract readonly id: RentalProviderId;
  abstract readonly configured: boolean;
  abstract quote(input: {
    minimumVramMb: number;
    spot: boolean;
    signal: AbortSignal;
  }): Promise<RentalQuote | null>;
  abstract create(input: {
    quote: RentalQuote;
    name: string;
    image: string;
    labels: Record<string, string>;
    environment: Record<string, string>;
    signal: AbortSignal;
  }): Promise<{ externalId: string }>;
  abstract terminate(externalId: string, signal: AbortSignal): Promise<void>;
  abstract exists(externalId: string, signal: AbortSignal): Promise<boolean>;

  protected async request(
    url: string,
    init: RequestInit,
    token: string | undefined,
  ): Promise<Response> {
    const response = await fetch(url, {
      ...init,
      headers: {
        accept: "application/json",
        ...(init.body ? { "content-type": "application/json" } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(init.headers ?? {}),
      },
    });
    if (!response.ok && response.status !== 404) {
      throw new Error(`${this.id}_http_${response.status}:${(await response.text()).slice(0, 300)}`);
    }
    return response;
  }
}

class GpuCloudRentalDriver extends JsonRentalDriver {
  readonly id = "gpu_cloud" as const;
  readonly configured: boolean;
  private readonly baseUrl: string;

  constructor(
    private readonly token: string | undefined,
    organizationId: string | undefined,
    projectId: string | undefined,
    private readonly maximumHourlyUsd: number | undefined,
  ) {
    super();
    this.configured = Boolean(token && organizationId && projectId && maximumHourlyUsd);
    this.baseUrl = `https://api.gpu_cloud.com/api/public/organizations/${encodeURIComponent(organizationId ?? "")}/projects/${encodeURIComponent(projectId ?? "")}/containers`;
  }

  async quote(input: { minimumVramMb: number; spot: boolean; signal: AbortSignal }) {
    if (input.spot) return null;
    return {
      offerId: "high",
      hourlyUsd: this.maximumHourlyUsd!,
      gpuModel: "provider-selected",
      vramMb: input.minimumVramMb,
      reliability: 0.99,
      verifiedHost: true,
      spot: false,
    };
  }

  async create(input: Parameters<RentalProviderDriver["create"]>[0]) {
    const response = await this.request(this.baseUrl, {
      method: "POST",
      signal: input.signal,
      body: JSON.stringify({
        name: input.name,
        display_name: input.name,
        replicas: 1,
        container: {
          image: input.image,
          environment_variables: input.environment,
          logging: { axiom: { host: "disabled" } },
        },
        resources: { gpu_classes: [input.quote.offerId], memory: 16384, cpu: 4 },
        restart_policy: "always",
        metadata: input.labels,
      }),
    }, this.token);
    const body = await response.json() as { id?: unknown; name?: unknown };
    const id = typeof body.id === "string" ? body.id : body.name;
    if (typeof id !== "string") throw new Error("gpu_cloud_create_missing_id");
    return { externalId: id };
  }

  async terminate(externalId: string, signal: AbortSignal) {
    await this.request(`${this.baseUrl}/${encodeURIComponent(externalId)}`, {
      method: "DELETE",
      signal,
    }, this.token);
  }

  async exists(externalId: string, signal: AbortSignal) {
    const response = await this.request(`${this.baseUrl}/${encodeURIComponent(externalId)}`, {
      method: "GET",
      signal,
    }, this.token);
    return response.status !== 404;
  }
}

class VastRentalDriver extends JsonRentalDriver {
  readonly id = "vast" as const;
  readonly configured: boolean;
  constructor(private readonly token: string | undefined) {
    super();
    this.configured = Boolean(token);
  }

  async quote(input: { minimumVramMb: number; spot: boolean; signal: AbortSignal }) {
    const query = encodeURIComponent(JSON.stringify({
      verified: { eq: true },
      reliability2: { gte: 0.98 },
      gpu_ram: { gte: input.minimumVramMb },
      rentable: { eq: true },
      rented: { eq: false },
      type: input.spot ? "bid" : "on-demand",
      order: [["dph_total", "asc"]],
      limit: 1,
    }));
    const response = await this.request(
      `https://console.vast.ai/api/v0/bundles/?q=${query}`,
      { method: "GET", signal: input.signal },
      this.token,
    );
    const body = await response.json() as { offers?: Array<Record<string, unknown>> };
    const offer = body.offers?.[0];
    if (!offer) return null;
    return {
      offerId: String(offer.id),
      hourlyUsd: Number(offer.dph_total),
      gpuModel: String(offer.gpu_name ?? "GPU"),
      vramMb: Number(offer.gpu_ram),
      reliability: Number(offer.reliability2),
      verifiedHost: offer.verified === true,
      spot: input.spot,
    };
  }

  async create(input: Parameters<RentalProviderDriver["create"]>[0]) {
    const response = await this.request(
      `https://console.vast.ai/api/v0/asks/${encodeURIComponent(input.quote.offerId)}/`,
      {
        method: "PUT",
        signal: input.signal,
        body: JSON.stringify({
          client_id: "me",
          image: input.image,
          disk: 80,
          label: input.name,
          onstart: "",
          env: input.environment,
          runtype: "args",
        }),
      },
      this.token,
    );
    const body = await response.json() as { new_contract?: unknown; id?: unknown };
    const id = body.new_contract ?? body.id;
    if (typeof id !== "string" && typeof id !== "number") throw new Error("vast_create_missing_id");
    return { externalId: String(id) };
  }

  async terminate(externalId: string, signal: AbortSignal) {
    await this.request(
      `https://console.vast.ai/api/v0/instances/${encodeURIComponent(externalId)}/`,
      { method: "DELETE", signal },
      this.token,
    );
  }

  async exists(externalId: string, signal: AbortSignal) {
    const response = await this.request(
      `https://console.vast.ai/api/v0/instances/${encodeURIComponent(externalId)}/`,
      { method: "GET", signal },
      this.token,
    );
    return response.status !== 404;
  }
}

class CloreRentalDriver extends JsonRentalDriver {
  readonly id = "clore" as const;
  readonly configured: boolean;
  constructor(private readonly token: string | undefined) {
    super();
    this.configured = Boolean(token);
  }

  async quote(input: { minimumVramMb: number; spot: boolean; signal: AbortSignal }) {
    const response = await this.request(
      "https://api.clore.ai/v1/marketplace",
      { method: "GET", signal: input.signal },
      this.token,
    );
    const body = await response.json() as { servers?: Array<Record<string, unknown>> };
    const offers = (body.servers ?? []).filter((offer) =>
      Number(offer.reliability ?? 0) >= 0.98
      && Number(offer.vram ?? offer.gpu_memory ?? 0) >= input.minimumVramMb
      && offer.verified !== false);
    offers.sort((left, right) => Number(left.price ?? Infinity) - Number(right.price ?? Infinity));
    const offer = offers[0];
    if (!offer) return null;
    return {
      offerId: String(offer.id),
      hourlyUsd: Number(offer.price),
      gpuModel: String(offer.gpu_name ?? "GPU"),
      vramMb: Number(offer.vram ?? offer.gpu_memory),
      reliability: Number(offer.reliability),
      verifiedHost: offer.verified !== false,
      spot: input.spot,
    };
  }

  async create(input: Parameters<RentalProviderDriver["create"]>[0]) {
    const response = await this.request(
      "https://api.clore.ai/v1/create_order",
      {
        method: "POST",
        signal: input.signal,
        body: JSON.stringify({
          server_id: input.quote.offerId,
          image: input.image,
          name: input.name,
          spot: input.quote.spot,
          env: input.environment,
          labels: input.labels,
        }),
      },
      this.token,
    );
    const body = await response.json() as { order_id?: unknown; id?: unknown };
    const id = body.order_id ?? body.id;
    if (typeof id !== "string" && typeof id !== "number") throw new Error("clore_create_missing_id");
    return { externalId: String(id) };
  }

  async terminate(externalId: string, signal: AbortSignal) {
    await this.request("https://api.clore.ai/v1/cancel_order", {
      method: "POST",
      signal,
      body: JSON.stringify({ order_id: externalId }),
    }, this.token);
  }

  async exists(externalId: string, signal: AbortSignal) {
    const response = await this.request(
      `https://api.clore.ai/v1/orders/${encodeURIComponent(externalId)}`,
      { method: "GET", signal },
      this.token,
    );
    return response.status !== 404;
  }
}

function timeoutController(milliseconds: number): AbortController {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error("provider_timeout")), milliseconds);
  timer.unref();
  controller.signal.addEventListener("abort", () => clearTimeout(timer), { once: true });
  return controller;
}
