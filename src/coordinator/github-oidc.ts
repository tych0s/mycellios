import { createRemoteJWKSet, jwtVerify, type JWTPayload } from "jose";

const GITHUB_ACTIONS_ISSUER = "https://token.actions.githubusercontent.com";
const RELEASE_UPLOAD_AUDIENCE = "mycellios-release-upload";
const RELEASE_REPOSITORY = "example/mycellios";
const RELEASE_REF = "refs/heads/main";
const RELEASE_WORKFLOW_POLICIES = [
  {
    workflowRef:
      `${RELEASE_REPOSITORY}/.github/workflows/node-build.yml@${RELEASE_REF}`,
    eventName: "push",
  },
  {
    workflowRef:
      `${RELEASE_REPOSITORY}/.github/workflows/publish-existing-release.yml@${RELEASE_REF}`,
    eventName: "workflow_dispatch",
  },
] as const;
const githubActionsKeys = createRemoteJWKSet(
  new URL(`${GITHUB_ACTIONS_ISSUER}/.well-known/jwks`),
);

export interface GitHubReleaseClaims {
  repository: string;
  ref: string;
  sha: string;
  workflowRef: string;
  eventName: "push" | "workflow_dispatch";
  environment: "production";
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
  const environment = stringClaim(payload, "environment");

  if (repository !== RELEASE_REPOSITORY) throw new Error("release_repository_not_allowed");
  if (ref !== RELEASE_REF) throw new Error("release_ref_not_allowed");
  const workflowPolicy = RELEASE_WORKFLOW_POLICIES.find(
    (candidate) => candidate.workflowRef === workflowRef,
  );
  if (!workflowPolicy) {
    throw new Error("release_workflow_not_allowed");
  }
  if (eventName !== workflowPolicy.eventName) throw new Error("release_event_not_allowed");
  if (environment !== "production") throw new Error("release_environment_not_allowed");
  if (!/^[0-9a-f]{40}$/.test(sha)) throw new Error("release_sha_invalid");

  return {
    repository,
    ref,
    sha,
    workflowRef,
    eventName: workflowPolicy.eventName,
    environment,
  };
}

function stringClaim(payload: JWTPayload, name: string): string {
  const value = payload[name];
  if (typeof value !== "string" || !value) throw new Error(`release_claim_missing_${name}`);
  return value;
}
