# Security policy

Mycellios coordinates untrusted, heterogeneous machines and can move model
activations between them. Security reports are therefore handled separately
from ordinary bug reports.

## Supported versions

Mycellios is currently pre-1.0. Security fixes are made on `main`; older
commits, development builds, research snapshots and unpublished artifacts are
not supported. When versioned releases begin, this table will be replaced by a
release support matrix.

## Reporting a vulnerability

Do not open a public issue, discussion or pull request for a suspected
vulnerability. Use GitHub's **Report a vulnerability** button on the repository
Security page to create a private security advisory. Include, when possible:

- the affected commit, component and platform;
- a minimal reproduction or proof of concept;
- the expected impact and required attacker position;
- whether credentials, model artifacts, activations or user data are exposed;
- any suggested remediation or embargo constraints.

Maintainers aim to acknowledge a report within three business days and provide
an initial assessment within seven. Remediation and disclosure timing depend on
severity, exploitability and the number of affected release channels. We will
coordinate public disclosure with the reporter after a fix or mitigation is
available.

## Scope

Reports are especially valuable for the coordinator, native worker and stage
runtime, launch/update agents, installers, authentication and allowlists,
transport protocols, model/artifact verification, release provenance, resource
isolation and denial-of-service boundaries.

The following are product security properties, not vulnerability claims:

- Link encryption protects data in transit, but a machine performing a model
  stage can observe the activations and weights it must process. Sensitive work
  must use private or trusted routes.
- A benchmark, simulator result or unverified hardware report is not evidence
  that a production path is safe or physically validated.
- Third-party model files and contributed compute remain untrusted inputs. A
  report that shows code execution, isolation escape, identity confusion,
  artifact substitution or unauthorized resource use is in scope.

Please avoid destructive testing against public services or machines you do not
own. Use synthetic data and the smallest safe proof of concept.

## Secrets and leaked credentials

If a credential appears in current code or Git history, treat it as compromised:
report it privately and rotate/revoke it. Removing the text from Git does not
invalidate the credential.
