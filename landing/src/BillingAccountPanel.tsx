import { CheckCircle2, CircleAlert, CreditCard, LoaderCircle, RefreshCw, ShieldCheck } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { AuthSession } from "./auth";
import {
  assertStripeRedirect,
  createBillingPortal,
  createSubscriptionCheckout,
  loadBillingAccount,
  newBillingIdempotencyKey,
  type BillingAccountOverview,
} from "./billing";

export function BillingAccountPanel({ session, titleId, navigate = (url) => window.location.assign(url) }: {
  session: AuthSession;
  titleId: string;
  navigate?: (url: string) => void;
}) {
  const [overview, setOverview] = useState<BillingAccountOverview | null>(null);
  const [busy, setBusy] = useState<"load" | "subscription" | "portal" | null>("load");
  const [error, setError] = useState<string | null>(null);
  const subscriptionKey = useRef<string | null>(null);
  const portalKey = useRef<string | null>(null);
  const verificationAttempts = useRef(0);
  const returnState = new URLSearchParams(window.location.search).get("checkout");

  const refresh = useCallback(async () => {
    setBusy("load");
    setError(null);
    try {
      setOverview(await loadBillingAccount(session.accessToken));
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(null);
    }
  }, [session.accessToken]);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    if (returnState !== "success" || overview?.subscription?.status === "active") return;
    if (!overview || verificationAttempts.current >= 6) return;
    verificationAttempts.current += 1;
    const timer = window.setTimeout(() => void refresh(), 2_500);
    return () => window.clearTimeout(timer);
  }, [overview, refresh, returnState]);

  async function subscribe() {
    setBusy("subscription");
    setError(null);
    subscriptionKey.current ??= newBillingIdempotencyKey("subscription");
    try {
      const result = await createSubscriptionCheckout(session.accessToken, subscriptionKey.current);
      subscriptionKey.current = null;
      navigate(assertStripeRedirect(result.url, "checkout"));
    } catch (caught) {
      setError(errorText(caught));
      setBusy(null);
    }
  }

  async function portal() {
    setBusy("portal");
    setError(null);
    portalKey.current ??= newBillingIdempotencyKey("portal");
    try {
      const result = await createBillingPortal(session.accessToken, portalKey.current);
      portalKey.current = null;
      navigate(assertStripeRedirect(result.url, "portal"));
    } catch (caught) {
      setError(errorText(caught));
      setBusy(null);
    }
  }

  if (busy === "load" && !overview) return <div className="billing-panel-state" role="status"><LoaderCircle className="spin" /><span>Loading verified billing state…</span></div>;
  if (!overview) return <div className="billing-panel-state error" role="alert"><CircleAlert /><span>{error ?? "Billing state is unavailable."}</span><button onClick={() => void refresh()}><RefreshCw />Retry</button></div>;

  const subscription = overview.subscription;
  const plan = overview.plan;
  const state = subscription?.status ?? "not_subscribed";
  const statusCopy = state === "active"
    ? { label: "ACTIVE", detail: `Renews through ${formatDate(subscription!.periodEnd)}.`, tone: "active" }
    : state === "past_due"
      ? { label: "PAYMENT DUE", detail: "Update the payment method in Stripe to keep access current.", tone: "warning" }
      : state === "cancelled"
        ? { label: "CANCELLED", detail: "The previous subscription is closed. You can start a new one.", tone: "muted" }
        : { label: "NOT SUBSCRIBED", detail: "Subscribe securely through Stripe. Access activates only after the signed payment webhook arrives.", tone: "muted" };

  return <>
    <h2 id={titleId}>Plan &amp; billing</h2>
    <p>Your subscription state comes from verified payment evidence, never from a browser redirect.</p>
    {returnState === "success" && state === "active" && <div className="billing-return-notice" role="status"><CheckCircle2 /><span><strong>Mycellios Go is active</strong><small>Your signed payment evidence is verified and the plan allowance is ready.</small></span></div>}
    {returnState === "success" && state !== "active" && <div className="billing-return-notice" role="status"><LoaderCircle className="spin" /><span><strong>Payment received by Stripe</strong><small>Waiting for signed activation evidence. Refresh in a moment.</small></span></div>}
    {returnState === "cancelled" && <div className="billing-return-notice neutral" role="status"><CircleAlert /><span><strong>Checkout closed</strong><small>No subscription was activated and no credits were granted.</small></span></div>}
    <section className="billing-plan-card" aria-label="Mycellios Go subscription">
      <header><div><span>MYCELLIOS GO</span><strong>{plan ? formatPrice(plan.amountMicros, plan.currency) : "€10"}<small>/month</small></strong></div><b className={`billing-status ${statusCopy.tone}`}>{state === "active" ? <CheckCircle2 /> : <ShieldCheck />}{statusCopy.label}</b></header>
      <p>{statusCopy.detail}</p>
      <div className="billing-plan-proof"><span><small>MONTHLY ACCESS</small><strong>{plan ? `${formatTokens(plan.includedTokens)} tokens` : "Configured in Stripe"}</strong></span><span><small>CURRENT BALANCE</small><strong>Visible in Account</strong></span></div>
      <footer>
        {state === "not_subscribed" || state === "cancelled" ? <button className="billing-primary" disabled={!overview.checkoutAvailable || busy !== null} onClick={() => void subscribe()}>{busy === "subscription" ? <LoaderCircle className="spin" /> : <CreditCard />}{busy === "subscription" ? "Opening Stripe…" : overview.checkoutAvailable ? "Subscribe with Stripe" : "Stripe sandbox unavailable"}</button> : null}
        {overview.portalAvailable && state !== "cancelled" ? <button className="billing-secondary" disabled={busy !== null} onClick={() => void portal()}>{busy === "portal" ? <LoaderCircle className="spin" /> : <CreditCard />}{busy === "portal" ? "Opening portal…" : "Manage billing"}</button> : null}
        <button className="billing-refresh" disabled={busy !== null} onClick={() => void refresh()} aria-label="Refresh billing state"><RefreshCw /></button>
      </footer>
    </section>
    {!overview.checkoutAvailable && <div className="billing-safety-note"><ShieldCheck /><span><strong>Payments are safely disabled</strong><small>The coordinator has no active Stripe Test configuration. No checkout or live charge can start.</small></span></div>}
    {overview.tokenDebt > 0 && <div className="billing-safety-note warning"><CircleAlert /><span><strong>{formatTokens(overview.tokenDebt)} token debt</strong><small>Future grants repay reversed usage before increasing the available balance.</small></span></div>}
    {error && <div className="account-auth-error" role="alert"><CircleAlert />{error}</div>}
  </>;
}

function formatPrice(amountMicros: number, currency: string): string {
  return new Intl.NumberFormat("en", { style: "currency", currency, maximumFractionDigits: 0 }).format(amountMicros / 1_000_000);
}

function formatTokens(tokens: number): string {
  return new Intl.NumberFormat("en", { notation: tokens >= 100_000 ? "compact" : "standard", maximumFractionDigits: 1 }).format(tokens);
}

function formatDate(timestamp: number): string {
  return new Intl.DateTimeFormat("en", { dateStyle: "medium" }).format(new Date(timestamp));
}

function errorText(value: unknown): string {
  return value instanceof Error ? value.message : "Billing request failed.";
}
