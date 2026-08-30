# Documentation

This directory contains current operating knowledge, not a diary of the
project. Start with the document that answers your question:

| Question | Document |
| --- | --- |
| What is the system and where are its boundaries? | [Architecture](ARCHITECTURE.md) |
| How do I attempt the first real two-machine run? | [Two-host quickstart](TWO_HOST_QUICKSTART.md) |
| What is implemented and what is actually proven? | [Status and evidence](STATUS_AND_EVIDENCE.md) |
| How do I change and verify the repository? | [Development](DEVELOPMENT.md) |
| Where does code belong and how will the monorepo evolve? | [Repository structure](REPOSITORY_STRUCTURE.md) |
| What are the runtime trust boundaries? | [Security](SECURITY.md) |
| What HTTP API does the coordinator expose? | [OpenAPI](openapi.yaml) |

## Source-of-truth rules

- The root `README.md` is the product entry point.
- `STATUS_AND_EVIDENCE.md` is the only human-readable capability status.
- A benchmark supports a claim only when its immutable artifact is committed
  and linked from the status document.
- Active implementation work lives in `specs/<feature>/`. Once complete, its
  lasting decisions are folded into these documents and the temporary spec is
  removed.
- Dated investigations, handoffs, competitive notes and projections are not
  product documentation.
