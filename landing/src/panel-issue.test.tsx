import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { AccountDetails } from "./AccountDetails";
import { AccountModal, Tests, friendlyModelMutationError, panelIssueForErrors, panelViewFromLocation } from "./Panel";

describe("Panel deep links", () => {
  it("opens the intended view for legacy links", () => {
    expect(panelViewFromLocation("/dashboard", "?view=network")).toBe("history");
    expect(panelViewFromLocation("/network", "?view=history")).toBe("history");
    expect(panelViewFromLocation("/dashboard", "?view=tasks")).toBe("jobs");
    expect(panelViewFromLocation("/dashboard", "?view=chat")).toBe("inference");
  });
});

describe("Panel issue classification", () => {
  it("does not label a local runtime failure as a coordinator outage", () => {
    expect(panelIssueForErrors(null, "Worker admission failed")).toEqual({
      source: "runtime",
      message: "Worker admission failed",
    });
  });

  it("keeps a real coordinator error authoritative", () => {
    expect(panelIssueForErrors("HTTP 503", "Worker admission failed")).toEqual({
      source: "coordinator",
      message: "HTTP 503",
    });
    expect(panelIssueForErrors(null, null)).toBeNull();
  });
});

describe("Network action errors", () => {
  it("explains that a timed-out mutation may still have completed", () => {
    expect(friendlyModelMutationError(new DOMException("The operation timed out", "TimeoutError")))
      .toBe("Request timed out. Check the current status before trying again.");
  });

  it("keeps administrator authorization actionable", () => {
    expect(friendlyModelMutationError(new Error("HTTP 401")))
      .toBe("The network administrator token is missing or invalid. Check it and try again.");
  });
});

describe("Benchmark controls", () => {
  it("keeps remote results readable without offering a local-only action", () => {
    const remote = renderToStaticMarkup(<Tests canRunLocally={false} />);
    const local = renderToStaticMarkup(<Tests canRunLocally={true} />);
    expect(remote).toContain("Manual tests can be started on the coordinator host.");
    expect(remote).not.toContain("Run test again");
    expect(local).toContain("Run test again");
  });
});

describe("AccountDetails", () => {
  it("does not offer linking providers known to be disabled", () => {
    const markup = renderToStaticMarkup(<AccountDetails
      tab="account"
      titleId="account-title"
      session={{ accessToken: "access", refreshToken: "refresh", expiresAt: Date.now() + 60_000, user: { id: "user-1", email: "person@example.com" } }}
      identity={null}
      account={null}
      usage={[]}
      usageBusy={false}
      linkBusy={null}
      roleBusy={false}
      providers={{ email: true, google: false, twitter: false }}
      error={null}
      onLinkProvider={() => undefined}
      onRetryRole={() => undefined}
      onSignOut={() => undefined}
    />);
    expect(markup).toContain("Sign-in methods");
    expect(markup).not.toContain(">Google</button>");
    expect(markup).not.toContain(">X</button>");
    expect(markup).toContain("No additional sign-in methods are available");
    expect(markup).not.toContain(">MetaMask</button>");
  });

  it("shows the signed-in account without granting an unverified role", () => {
    const markup = renderToStaticMarkup(<AccountDetails
      tab="account"
      titleId="account-title"
      session={{ accessToken: "access", refreshToken: "refresh", expiresAt: Date.now() + 60_000, user: { id: "user-1", email: "person@example.com" } }}
      identity={null}
      account={null}
      usage={[]}
      usageBusy={false}
      linkBusy={null}
      roleBusy={false}
      error={null}
      onLinkProvider={() => undefined}
      onRetryRole={() => undefined}
      onSignOut={() => undefined}
    />);
    expect(markup).toContain("person@example.com");
    expect(markup).toContain("Network role unverified");
    expect(markup).toContain("Admin tools remain locked.");
    expect(markup).toContain("Retry");
    expect(markup).toContain("REQUESTS MADE</small><strong>—</strong>");
  });

  it("shows usage errors instead of reporting missing data as zero activity", () => {
    const markup = renderToStaticMarkup(<AccountDetails
      tab="usage"
      titleId="usage-title"
      session={{ accessToken: "access", refreshToken: "refresh", expiresAt: Date.now() + 60_000, user: { id: "user-1", email: null } }}
      identity={null}
      account={null}
      usage={[]}
      usageBusy={false}
      linkBusy={null}
      roleBusy={false}
      error="HTTP 503"
      onLinkProvider={() => undefined}
      onRetryRole={() => undefined}
      onSignOut={() => undefined}
    />);
    expect(markup).toContain("Usage could not be loaded: HTTP 503");
    expect(markup).toContain("LAST 50 REQUESTS BY MODEL");
    expect(markup).not.toContain("No usage recorded yet.");
    expect(markup).toContain("REQUESTS</small><strong>—</strong>");
  });
});

describe("AccountModal", () => {
  it("waits for session restoration before presenting sign-in controls", () => {
    const markup = renderToStaticMarkup(<AccountModal
      ready={false}
      config={{ enabled: false }}
      onConfigLoaded={() => undefined}
      getValidSession={async () => null}
      session={null}
      identity={null}
      account={null}
      onAuthenticated={() => undefined}
      onRetryRole={async () => undefined}
      onSignOut={() => undefined}
      onClose={() => undefined}
    />);
    expect(markup).toContain("Checking your account");
    expect(markup).not.toContain("Sign in securely");
  });

  it("waits for provider availability before offering Google or X", () => {
    const markup = renderToStaticMarkup(<AccountModal
      config={{ enabled: false }}
      onConfigLoaded={() => undefined}
      getValidSession={async () => null}
      session={null}
      identity={null}
      account={null}
      onAuthenticated={() => undefined}
      onRetryRole={async () => undefined}
      onSignOut={() => undefined}
      onClose={() => undefined}
    />);
    expect(markup).not.toContain("Continue with Google");
    expect(markup).not.toContain("Continue with X");
    expect(markup).not.toContain("Continue with MetaMask");
    expect(markup).not.toContain("or continue with");
  });

  it("offers only available identity providers", () => {
    const markup = renderToStaticMarkup(<AccountModal
      config={{ enabled: true, url: "https://auth.example", anonKey: "anon", providers: { email: false, google: false, twitter: true } }}
      onConfigLoaded={() => undefined}
      getValidSession={async () => null}
      session={null}
      identity={null}
      account={null}
      onAuthenticated={() => undefined}
      onRetryRole={async () => undefined}
      onSignOut={() => undefined}
      onClose={() => undefined}
    />);
    expect(markup).toContain("Email sign-in is unavailable");
    expect(markup).not.toContain('type="email"');
    expect(markup).not.toContain("Continue with Google");
    expect(markup).toContain("Continue with X");
    expect(markup).not.toContain("Continue with MetaMask");
  });

  it("offers wallet sign-in only after explicit Web3 availability", () => {
    const markup = renderToStaticMarkup(<AccountModal
      config={{ enabled: true, url: "https://auth.example", anonKey: "anon", providers: { email: true, google: false, twitter: false, web3: true } }}
      onConfigLoaded={() => undefined}
      getValidSession={async () => null}
      session={null}
      identity={null}
      account={null}
      onAuthenticated={() => undefined}
      onRetryRole={async () => undefined}
      onSignOut={() => undefined}
      onClose={() => undefined}
    />);
    expect(markup).toContain("Continue with MetaMask");
  });
});
