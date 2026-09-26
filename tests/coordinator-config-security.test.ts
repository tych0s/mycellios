import { generateKeyPairSync } from "node:crypto";
import { describe, expect, it } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  assertCoordinatorNetworkSecurity,
  isCoordinatorLoopbackHost,
  loadCoordinatorConfig,
} from "../src/core/config.js";

describe("coordinator network configuration security", () => {
  it.each([
    "localhost",
    "127.0.0.1",
    "127.25.10.3",
    "::1",
    "[::1]",
    "0:0:0:0:0:0:0:1",
    "::ffff:127.0.0.1",
  ])("recognizes loopback host %s", (host) => {
    expect(isCoordinatorLoopbackHost(host)).toBe(true);
    expect(() => assertCoordinatorNetworkSecurity({ host })).not.toThrow();
  });

  it.each([
    "0.0.0.0",
    "::",
    "[::]",
    "192.168.1.20",
    "10.0.0.4",
    "127.example.com",
    "coordinator.internal",
  ])("rejects unauthenticated non-loopback host %s", (host) => {
    expect(isCoordinatorLoopbackHost(host)).toBe(false);
    expect(() => assertCoordinatorNetworkSecurity({ host })).toThrow(
      "MYCELLIOS_NETWORK_TOKEN is required when GPU_MESH_HOST is not loopback.",
    );
  });

  it("keeps the default local coordinator available without a network token", () => {
    const config = loadCoordinatorConfig({ GPU_MESH_DB: ":memory:" });
    expect(config.host).toBe("127.0.0.1");
    expect(config.networkToken).toBeUndefined();
  });

  it("separates the browser Supabase origin from the internal server origin", () => {
    const environment = {
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SUPABASE_URL: "http://supabase-internal:8443",
      MYCELLIOS_SUPABASE_SERVICE_ROLE_KEY: "service-role",
      MYCELLIOS_SUPABASE_ANON_KEY: "public-anon",
    };
    expect(loadCoordinatorConfig({
      ...environment,
      MYCELLIOS_SUPABASE_PUBLIC_URL: "https://auth.example/",
    })).toMatchObject({
      supabaseUrl: "http://supabase-internal:8443",
      publicSupabaseUrl: "https://auth.example",
    });
    expect(() => loadCoordinatorConfig({
      ...environment,
      MYCELLIOS_SUPABASE_PUBLIC_URL: "http://auth.example",
    })).toThrow("MYCELLIOS_SUPABASE_PUBLIC_URL must be an HTTPS origin");
    expect(() => loadCoordinatorConfig({
      ...environment,
      MYCELLIOS_SUPABASE_PUBLIC_URL: "not-a-url",
    })).toThrow("MYCELLIOS_SUPABASE_PUBLIC_URL must be an HTTPS origin");
  });

  it("allows a public bind only when the network token is present", () => {
    const config = loadCoordinatorConfig({
      GPU_MESH_HOST: " 0.0.0.0 ",
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_NETWORK_TOKEN: " network-secret ",
    });
    expect(config.host).toBe("0.0.0.0");
    expect(config.networkToken).toBe("network-secret");
  });

  it("rejects a public bind during environment configuration loading", () => {
    expect(() => loadCoordinatorConfig({
      GPU_MESH_HOST: "0.0.0.0",
      GPU_MESH_DB: ":memory:",
    })).toThrow(
      "MYCELLIOS_NETWORK_TOKEN is required when GPU_MESH_HOST is not loopback.",
    );
  });

  it("requires a complete server-side Stripe Checkout configuration", () => {
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_STRIPE_SECRET_KEY: "sk_test_1234567890abcdef",
    })).toThrow("Stripe Checkout requires");

    const config = loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_STRIPE_SECRET_KEY: "sk_test_1234567890abcdef",
      MYCELLIOS_STRIPE_GO_PRICE_ID: "price_go_1",
      MYCELLIOS_GO_INCLUDED_TOKENS: "50000",
      MYCELLIOS_BILLING_SUCCESS_URL: "https://mycellios.com/account?checkout=success",
      MYCELLIOS_BILLING_CANCEL_URL: "https://mycellios.com/account?checkout=cancelled",
      MYCELLIOS_BILLING_PORTAL_RETURN_URL: "https://mycellios.com/account",
      MYCELLIOS_BILLING_TOPUP_PACKS_JSON: JSON.stringify([{
        packId: "boost-5",
        amountMicros: 5_000_000,
        currency: "EUR",
        tokenAmount: 20_000,
        stripePriceId: "price_boost_5",
      }]),
    });
    expect(config.stripeCheckout).toMatchObject({
      subscriptionPriceId: "price_go_1",
      includedTokens: 50_000,
      topUpPacks: [{ packId: "boost-5", tokenAmount: 20_000 }],
    });
  });

  it("loads stablecoin watcher trust and finality policy from configuration", () => {
    const publicKey = generateKeyPairSync("ed25519").publicKey
      .export({ type: "spki", format: "pem" }).toString();
    const config = loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_STABLECOIN_WATCHER_CONFIG_JSON: JSON.stringify({
        trustedWatcherKeys: [{ keyId: "watcher-primary", publicKey }],
        chains: [{ chainId: "base-mainnet", asset: "USDC", minimumConfirmations: 20 }],
      }),
    });
    expect(config.stablecoinWatcher).toMatchObject({
      trustedWatcherKeys: [{ keyId: "watcher-primary" }],
      chains: [{ chainId: "base-mainnet", asset: "USDC", minimumConfirmations: 20 }],
    });
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_STABLECOIN_WATCHER_CONFIG_JSON: "[]",
    })).toThrow("must contain an object");
  });

  it("loads seller payout policy and verifier keys as one fail-closed bundle", () => {
    const publicKey = generateKeyPairSync("ed25519").publicKey
      .export({ type: "spki", format: "pem" }).toString();
    const config = loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON: JSON.stringify({
        minimumUsdMicros: 10_000_000,
        spore: {
          legalApproved: false,
          custodyApproved: false,
          liquidityApproved: false,
          antifraudApproved: false,
        },
        destinationVerifierKeys: [{ keyId: "payout-kyc-primary", publicKey }],
        settlementVerifierKeys: [{ keyId: "payout-settlement-primary", publicKey }],
        settlementEvidenceMaxAgeMs: 900_000,
      }),
    });
    expect(config.sellerPayout).toMatchObject({
      minimumUsdMicros: 10_000_000,
      spore: { legalApproved: false, antifraudApproved: false },
      destinationVerifierKeys: [{ keyId: "payout-kyc-primary" }],
      settlementVerifierKeys: [{ keyId: "payout-settlement-primary" }],
      settlementEvidenceMaxAgeMs: 900_000,
    });
    const payoutBundle = JSON.stringify({
      minimumUsdMicros: 10_000_000,
      spore: {
        legalApproved: false, custodyApproved: false,
        liquidityApproved: false, antifraudApproved: false,
      },
      destinationVerifierKeys: [{ keyId: "payout-kyc-primary", publicKey }],
      settlementVerifierKeys: [{ keyId: "payout-settlement-primary", publicKey }],
      settlementEvidenceMaxAgeMs: 900_000,
    });
    expect(loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON: payoutBundle,
      MYCELLIOS_STRIPE_CONNECT_SECRET_KEY: "sk_test_1234567890abcdef",
      MYCELLIOS_STRIPE_CONNECT_API_VERSION: "2025-10-29.clover",
    }).stripeConnectPayout).toMatchObject({ apiVersion: "2025-10-29.clover" });
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON: payoutBundle,
      MYCELLIOS_STRIPE_CONNECT_SECRET_KEY: "sk_test_1234567890abcdef",
    })).toThrow("and MYCELLIOS_STRIPE_CONNECT_API_VERSION together");

    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON: JSON.stringify({
        minimumUsdMicros: 10_000_000,
        spore: { legalApproved: false },
        destinationVerifierKeys: [{ keyId: "payout-kyc-primary", publicKey }],
        settlementVerifierKeys: [{ keyId: "payout-settlement-primary", publicKey }],
        settlementEvidenceMaxAgeMs: 900_000,
      }),
    })).toThrow("all SPORE gates");
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON: JSON.stringify({
        minimumUsdMicros: 10_000_000,
        spore: {
          legalApproved: false, custodyApproved: false,
          liquidityApproved: false, antifraudApproved: false,
        },
        destinationVerifierKeys: [],
        settlementVerifierKeys: [{ keyId: "payout-settlement-primary", publicKey }],
        settlementEvidenceMaxAgeMs: 900_000,
      }),
    })).toThrow("at least one destination verifier key");
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON: JSON.stringify({
        minimumUsdMicros: 10_000_000,
        spore: {
          legalApproved: false, custodyApproved: false,
          liquidityApproved: false, antifraudApproved: false,
        },
        destinationVerifierKeys: [{ keyId: "bad-key", publicKey: "x".repeat(64) }],
        settlementVerifierKeys: [{ keyId: "payout-settlement-primary", publicKey }],
        settlementEvidenceMaxAgeMs: 900_000,
      }),
    })).toThrow("valid Ed25519 public keys");
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON: JSON.stringify({
        minimumUsdMicros: 10_000_000,
        spore: {
          legalApproved: false, custodyApproved: false,
          liquidityApproved: false, antifraudApproved: false,
        },
        destinationVerifierKeys: [{ keyId: "payout-kyc-primary", publicKey }],
        settlementVerifierKeys: [],
        settlementEvidenceMaxAgeMs: 900_000,
      }),
    })).toThrow("at least one settlement verifier key");
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON: JSON.stringify({
        minimumUsdMicros: 10_000_000,
        spore: {
          legalApproved: false, custodyApproved: false,
          liquidityApproved: false, antifraudApproved: false,
        },
        destinationVerifierKeys: [{ keyId: "payout-kyc-primary", publicKey }],
        settlementVerifierKeys: [{ keyId: "payout-settlement-primary", publicKey }],
        settlementEvidenceMaxAgeMs: 30_000,
      }),
    })).toThrow("between 60000 and 86400000");

    const sporeBundle = JSON.parse(payoutBundle) as Record<string, unknown>;
    sporeBundle.spore = {
      legalApproved: true, custodyApproved: true,
      liquidityApproved: true, antifraudApproved: true,
    };
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON: JSON.stringify(sporeBundle),
    })).toThrow("requires a complete sporeConversion policy");
    sporeBundle.sporeConversion = {
      trustedOracleKeys: [{ keyId: "spore-oracle-primary", publicKey }],
      approvedAssets: [{ chainId: "base", assetId: "spore-contract-1", tokenDecimals: 18 }],
      maxQuoteAgeMs: 300_000,
    };
    expect(loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON: JSON.stringify(sporeBundle),
    }).sellerPayout?.sporeConversion).toMatchObject({
      approvedAssets: [{ chainId: "base", tokenDecimals: 18 }],
      trustedOracleKeys: [{ keyId: "spore-oracle-primary" }],
    });
  });
  it("loads an Ed25519 component-update keyring for safe key rotation", () => {
    const root = mkdtempSync(join(tmpdir(), "mycellios-keyring-"));
    try {
      const spki = generateKeyPairSync("ed25519").publicKey.export({
        format: "der",
        type: "spki",
      }).toString("base64url");
      const keyring = join(root, "dev-keys.json");
      writeFileSync(keyring, JSON.stringify([
        { keyId: "mycellios-dev-old", spki },
        { keyId: "mycellios-dev-new", spki },
      ]));

      const config = loadCoordinatorConfig({
        GPU_MESH_DB: ":memory:",
        MYCELLIOS_COMPONENT_UPDATE_DEV_KEYRING_FILE: keyring,
      });

      expect(config.componentUpdatePinnedKeys?.dev).toEqual([
        { keyId: "mycellios-dev-old", spki },
        { keyId: "mycellios-dev-new", spki },
      ]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects component-update data that is base64url but not Ed25519 SPKI", () => {
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_COMPONENT_UPDATE_DEV_KEY_ID: "mycellios-dev-invalid",
      MYCELLIOS_COMPONENT_UPDATE_DEV_PUBLIC_KEY:
        Buffer.from("not-an-spki").toString("base64url"),
    })).toThrow("Invalid dev component update public key configuration.");
  });

  it("loads model-certification verification only from a complete Ed25519 pin", () => {
    const spki = generateKeyPairSync("ed25519").publicKey.export({ format: "der", type: "spki" }).toString("base64url");
    expect(loadCoordinatorConfig({ GPU_MESH_DB: ":memory:", MYCELLIOS_MODEL_CERTIFICATION_KEY_ID: "cert-1",
      MYCELLIOS_MODEL_CERTIFICATION_PUBLIC_KEY: spki }).modelCertificationPinnedKeys).toEqual([{ keyId: "cert-1", spki }]);
    expect(() => loadCoordinatorConfig({ GPU_MESH_DB: ":memory:", MYCELLIOS_MODEL_CERTIFICATION_KEY_ID: "cert-1" }))
      .toThrow("must be configured together");
    expect(() => loadCoordinatorConfig({ GPU_MESH_DB: ":memory:", MYCELLIOS_MODEL_CERTIFICATION_KEY_ID: "cert-1",
      MYCELLIOS_MODEL_CERTIFICATION_PUBLIC_KEY: Buffer.from("not-spki").toString("base64url") }))
      .toThrow("Invalid model certification public key configuration");
  });

  it("requires the economic receipt key id and private key together", () => {
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_ECONOMIC_RECEIPT_SIGNING_KEY_ID: "economic-prod-1",
    })).toThrow("must be configured together");
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_ECONOMIC_RECEIPT_SIGNING_PRIVATE_KEY: "private-key-only",
    })).toThrow("must be configured together");
  });

  it("loads the economic receipt signing key from a secret file", () => {
    const root = mkdtempSync(join(tmpdir(), "mycellios-economic-key-"));
    try {
      const privateKey = generateKeyPairSync("ed25519").privateKey.export({
        format: "pem",
        type: "pkcs8",
      }).toString();
      const keyFile = join(root, "economic-signing-key.pem");
      writeFileSync(keyFile, privateKey, { mode: 0o600 });

      const config = loadCoordinatorConfig({
        GPU_MESH_DB: ":memory:",
        MYCELLIOS_ECONOMIC_RECEIPT_SIGNING_KEY_ID: "economic-prod-1",
        MYCELLIOS_ECONOMIC_RECEIPT_SIGNING_PRIVATE_KEY_FILE: keyFile,
      });

      expect(config.economicReceiptSigningKeyId).toBe("economic-prod-1");
      expect(config.economicReceiptSigningPrivateKey).toBe(privateKey.trim());
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
