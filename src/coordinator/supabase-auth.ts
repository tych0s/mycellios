const PUBLIC_NETWORK_ID = "00000000-0000-0000-0000-000000000001";

export type NetworkRole = "owner" | "admin" | "operator" | "viewer";

export interface AuthenticatedNetworkUser {
  id: string;
  email: string | null;
  role: NetworkRole | null;
}

export class SupabaseAuthService {
  private readonly baseUrl: URL;

  constructor(
    url: string,
    private readonly serviceRoleKey: string,
    private readonly fetchImpl: typeof fetch = fetch,
  ) {
    this.baseUrl = new URL(url);
  }

  async authenticate(accessToken: string): Promise<AuthenticatedNetworkUser | null> {
    const userResponse = await this.fetchImpl(new URL("/auth/v1/user", this.baseUrl), {
      method: "GET",
      signal: AbortSignal.timeout(10_000),
      headers: {
        apikey: this.serviceRoleKey,
        authorization: `Bearer ${accessToken}`,
      },
    });
    if (userResponse.status === 401 || userResponse.status === 403) return null;
    if (!userResponse.ok) {
      throw new Error(`Supabase Auth returned HTTP ${userResponse.status}.`);
    }
    const user = await userResponse.json() as { id?: unknown; email?: unknown };
    if (typeof user.id !== "string") return null;
    const roleResponse = await this.fetchImpl(
      new URL(
        `/rest/v1/network_members?network_id=eq.${PUBLIC_NETWORK_ID}`
        + `&user_id=eq.${encodeURIComponent(user.id)}&select=role&limit=1`,
        this.baseUrl,
      ),
      {
        method: "GET",
        signal: AbortSignal.timeout(10_000),
        headers: {
          apikey: this.serviceRoleKey,
          authorization: `Bearer ${this.serviceRoleKey}`,
        },
      },
    );
    if (!roleResponse.ok) {
      throw new Error(`Supabase membership lookup returned HTTP ${roleResponse.status}.`);
    }
    const rows = await roleResponse.json() as Array<{ role?: unknown }>;
    const role = rows[0]?.role;
    return {
      id: user.id,
      email: typeof user.email === "string" ? user.email : null,
      role: isNetworkRole(role) ? role : null,
    };
  }
}

function isNetworkRole(value: unknown): value is NetworkRole {
  return value === "owner" || value === "admin" || value === "operator" || value === "viewer";
}
