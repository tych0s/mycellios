import { createRemoteJWKSet, jwtVerify, type JWTPayload } from "jose";

const GITHUB_ACTIONS_ISSUER = "https://token.actions.githubusercontent.com";
const RELEASE_UPLOAD_AUDIENCE = "mycellios-release-upload";
const RELEASE_REPOSITORY = "tych0s/mycellios";
const RELEASE_WORKFLOW = `${RELEASE_REPOSITORY}/.github/workflows/desktop-build.yml@`;
const githubActionsKeys = createRemoteJWKSet(
  new URL(`${GITHUB_ACTIONS_ISSUER}/.well-known/jwks`),
);

export interface GitHubReleaseClaims {
  repository: string;
  ref: string;
  sha: string;
  workflowRef: string;
}

export async function verifyGitHubReleaseUploadToken(
  token: string,
): Promise<GitHubReleaseClaims> {
  const { payload } = await jwtVerify(token, githubActionsKeys, {
    issuer: GITHUB_ACTIONS_ISSUER,
    audience: RELEASE_UPLOAD_AUDIENCE,
    algorithms: ["RS256"],
    clockTolerance: 5,
    maxTokenAge: "10m",
  });
  return validateGitHubReleaseClaims(payload);
}

export function validateGitHubReleaseClaims(payload: JWTPayload): GitHubReleaseClaims {
  const repository = stringClaim(payload, "repository");
  const ref = stringClaim(payload, "ref");
  const sha = stringClaim(payload, "sha");
  const workflowRef = stringClaim(payload, "workflow_ref");
  const eventName = stringClaim(payload, "event_name");

  if (repository !== RELEASE_REPOSITORY) throw new Error("release_repository_not_allowed");
  if (eventName !== "push") throw new Error("release_event_not_allowed");
  if (ref !== "refs/heads/main" && !/^refs\/tags\/v\d+\.\d+\.\d+$/.test(ref)) {
    throw new Error("release_ref_not_allowed");
  }
  if (!workflowRef.startsWith(RELEASE_WORKFLOW)) {
    throw new Error("release_workflow_not_allowed");
  }
  if (!/^[0-9a-f]{40}$/.test(sha)) throw new Error("release_sha_invalid");

  return { repository, ref, sha, workflowRef };
}

function stringClaim(payload: JWTPayload, name: string): string {
  const value = payload[name];
  if (typeof value !== "string" || !value) throw new Error(`release_claim_missing_${name}`);
  return value;
}
