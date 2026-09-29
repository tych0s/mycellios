# Native installer downloads

## Product gate

The public download link must deliver the same native installer family that
the verified node build produces: MSI for Windows x64, PKG for macOS arm64 and
DEB for Linux x64. A successful CI artifact upload alone does not make a
package available at `/downloads/windows`, `/downloads/macos-arm64` or
`/downloads/linux`.

## Requirements

- Each public release transaction binds all three installers to one source
  revision, source ID and version, with exact sizes and SHA-256 digests.
- Assemble the three CI artifact trees into one candidate directory only after
  checking each staged payload and provenance, exact source identity, package
  digest and matching CI run/attempt. Generate the update feed and checksum
  list from those checked bytes; reject mixed or modified artifacts.
- The coordinator serves a committed installer only after the full transaction
  passes validation. A failed upload or rollback cannot leave one platform
  pointing to a different release.
- Stable platform URLs redirect to versioned native filenames. The availability
  API and landing page show MSI, PKG and DEB accurately.
- Preserve access to any previously committed archive transaction during
  rollback. Do not silently reinterpret an existing ZIP or tarball as a native
  installer.
- The Windows USB launcher continues to consume a verified MSI and matching
  checksum. A downloaded MSI must pass the same package verification before
  it is given to a tester.
- No public capability claim is made until the published URL is fetched,
  checksum/provenance is verified, and a first installation is exercised.

## Verification

- Focused transaction, CLI, public route and landing tests cover native names,
  formats, atomic publication, rollback and missing artifacts.
- The CI native build produces the three package types and their evidence.
- Candidate assembly is atomic and tested with three synthetic CI job trees,
  including mixed-run and tampered-package rejection. The evidence signature
  state is metadata, not a cryptographic signature verification or promotion.
- The native CI workflow downloads its three matrix artifacts and uploads one
  assembled candidate plus a sealed transaction manifest; this job does not
  publish or sign them. A real CI run is still required to verify the workflow.
- Fetch the public package after publication and compare it with the CI
  artifact hash before counting it as available to a tester.
