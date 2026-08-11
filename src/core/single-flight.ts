export class SingleFlight {
  private pending: Promise<void> | null = null;

  run(operation: () => Promise<void>): Promise<void> {
    if (this.pending) return this.pending;
    const pending = Promise.resolve().then(operation);
    this.pending = pending;
    void pending.finally(() => {
      if (this.pending === pending) this.pending = null;
    }).catch(() => undefined);
    return pending;
  }

  wait(): Promise<void> {
    return this.pending ?? Promise.resolve();
  }
}
