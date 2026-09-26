import { CheckCircle2, CircleAlert, CreditCard, LoaderCircle, RefreshCw, ShieldCheck } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiAccount } from "./api-access";
import type { AuthSession } from "./auth";
import { formatCompactTokens } from "./panel-ui-utilities";
import {
  assertStripeRedirect,
  createBillingPortal,
  createSubscriptionCheckout,
  loadBillingAccount,
  newBillingIdempotencyKey,
  type BillingAccountOverview,
} from "./billing";

export function BillingAccountPanel({ session, titleId, account, navigate = (url) => window.location.assign(url) }: {
  session: AuthSession;
  titleId: string;
  account?: Pick<ApiAccount, "token_balance"> | null;
  navigate?: (url: string) => void;
}) {
  const [overview, setOverview] = useState<BillingAccountOverview | null>(null);
  const [busy, setBusy] = useState<"load" | "subscription" | "portal" | null>("load");
  const [error, setError] = useState<string | null>(null);
  const subscriptionKey = useRef<string | null>(null);
  const portalKey = useRef<string | null>(null);
  const [verificationAttempts, setVerificationAttempts] = useState(0);
  const refreshRequestSeq = useRef(0);
  const returnState = new URLSearchParams(window.location.search).get("checkout");

  const refresh = useCallback(async () => {
    const requestSeq = ++refreshRequestSeq.current;
    setBusy("load");
    setError(null);
    try {
      const next = await loadBillingAccount(session.accessToken);
      if (refreshRequestSeq.current === requestSeq) setOverview(next);
    } catch (caught) {
      if (refreshRequestSeq.current === requestSeq) setError(errorText(caught));
    } finally {
      if (refreshRequestSeq.current === requestSeq) setBusy(null);
    }
  }, [session.accessToken]);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    if (returnState !== "success" || !overview || overview.subscription?.status === "active" || busy !== null || verificationAttempts >= 6) return;
    const timer = window.setTimeout(() => {
      setVerificationAttempts((current) => current + 1);
      void refresh();
    }, 2_500);
    return () => window.clearTimeout(timer);
  }, [Boolean(overview), overview?.subscription?.status, busy, refresh, returnState, verificationAttempts]);

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
  const checkoutReady = overview.checkoutAvailable && plan?.status === "active";
  const checkoutChecking = verificationAttempts < 6 || busy === "load";
  const state = subscription?.status ?? "not_subscribed";
  const statusCopy = state === "active"
    ? { label: "ACTIVE", detail: `Renews through ${formatDate(subscription!.periodEnd)}.`, tone: "active" }
    : state === "past_due"
      ? { label: "PAYMENT DUE", detail: "Update the payment method in Stripe to keep access current.", tone: "warning" }
      : state === "cancelled"
        ? { label: "CANCELLED", detail: checkoutReady ? "The previous subscription is closed. You can start a new one." : "The previous subscription is closed. Checkout is unavailable right now.", tone: "muted" }
        : { label: "NOT SUBSCRIBED", detail: returnState === "success" ? "Checkout is being verified. Access remains unchanged until payment is confirmed." : checkoutReady ? "Subscribe securely through Stripe. Access activates after payment is confirmed." : "Subscription checkout is unavailable right now.", tone: "muted" };

  return <>
    <h2 id={titleId}>Plan &amp; billing</h2>
    <p>Your plan updates after Stripe confirms your payment.</p>
    {returnState === "success" && state === "active" && <div className="billing-return-notice" role="status"><CheckCircle2 /><span><strong>Mycellios Go is active</strong><small>Your payment is confirmed and your monthly tokens are ready.</small></span></div>}
    {returnState === "success" && state !== "active" && <div className={`billing-return-notice${checkoutChecking ? "" : " neutral"}`} role="status">{checkoutChecking ? <LoaderCircle className="spin" /> : <CircleAlert />}<span><strong>{checkoutChecking ? "Checking checkout status" : "Checkout is still unverified"}</strong><small>{checkoutChecking ? "Access activates after the payment provider confirms this checkout." : "Automatic checks have finished. Refresh the billing state later before starting another checkout."}</small></span></div>}
    {returnState === "cancelled" && state !== "active" && <div className="billing-return-notice neutral" role="status"><CircleAlert /><span><strong>Checkout closed</strong><small>Current billing status is shown below.</small></span></div>}
    <section className="billing-plan-card" aria-label="Mycellios Go subscription">
      <header><div><span>MYCELLIOS GO</span><strong className={plan?.status === "active" ? undefined : "price-unavailable"}>{billingPriceLabel(plan)}{plan?.status === "active" && <small>/month</small>}</strong></div><b className={`billing-status ${statusCopy.tone}`}>{state === "active" ? <CheckCircle2 /> : <ShieldCheck />}{statusCopy.label}</b></header>
      <p>{statusCopy.detail}</p>
      <div className="billing-plan-proof"><span><small>INCLUDED EACH MONTH</small><strong>{plan?.status === "active" ? `${formatTokens(plan.includedTokens)} tokens` : "No active plan"}</strong></span><span><small>ACCOUNT BALANCE</small><strong>{account ? `${formatCompactTokens(account.token_balance)} tokens` : "Unavailable"}</strong></span></div>
      <footer>
        {state === "not_subscribed" || state === "cancelled" ? <button className="billing-primary" disabled={!checkoutReady || busy !== null} onClick={() => void subscribe()}>{busy === "subscription" ? <LoaderCircle className="spin" /> : <CreditCard />}{busy === "subscription" ? "Opening Stripe…" : checkoutReady ? "Subscribe with Stripe" : plan?.status !== "active" ? "Plan unavailable" : "Checkout unavailable"}</button> : null}
        {overview.portalAvailable && state !== "cancelled" ? <button className="billing-secondary" disabled={busy !== null} onClick={() => void portal()}>{busy === "portal" ? <LoaderCircle className="spin" /> : <CreditCard />}{busy === "portal" ? "Opening portal…" : "Manage billing"}</button> : null}
        <button className="billing-refresh" disabled={busy !== null} onClick={() => void refresh()} aria-label="Refresh billing state"><RefreshCw /></button>
      </footer>
    </section>
    {!checkoutReady && <div className="billing-safety-note"><ShieldCheck /><span><strong>Checkout unavailable</strong><small>{plan?.status !== "active" ? "No active subscription plan is available right now." : "Payment checkout is not available for this account right now."}</small></span></div>}
    {overview.tokenDebt > 0 && <div className="billing-safety-note warning"><CircleAlert /><span><strong>{formatTokens(overview.tokenDebt)} token debt</strong><small>Future grants repay reversed usage before increasing the available balance.</small></span></div>}
    {error && <div className="account-auth-error" role="alert"><CircleAlert />{error}</div>}
  </>;
}

export function billingPriceLabel(plan: BillingAccountOverview["plan"]): string {
  return plan?.status === "active" ? formatPrice(plan.amountMicros, plan.currency) : "Price unavailable";
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
