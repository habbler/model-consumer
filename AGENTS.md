# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the pure domain modules. Start with `Core` (types, dynamics) and `Runner` scaffolding; keep IO at the boundaries. Existing scaffold `MyLib.hs` will be replaced as phases land.
- `app/` contains executable entry points (default: `Main.hs`). Extend this with scenario selection and runner wiring.
- `test/` mirrors `src/` structure using `hspec`, and should include property tests plus golden fixtures as subsystems mature.
- `docs/` holds specifications (`Specification.md`, `ImplementationPlan.md`) and design notes. Update these alongside structural changes.
- `ProjectStatus.md` in repo root tracks execution progress; refresh it after completing milestones or shifting priorities.

## Build, Test, and Development Commands
- `cabal build` – compile the library and executables; run after dependency or interface changes.
- `cabal test` – execute the test suite; required before any PR.
- `cabal run model-consumer` – launch the CLI (extend via `app/Main.hs` when runners are wired).
- `ghcid --command="cabal repl model-consumer"` – optional rapid dev loop with live reload.
- Web search is available in the CLI; use it to source Hasktorch docs or academic references before updating models or specs.

## Coding Style & Naming Conventions
- Haskell modules use explicit export lists, four-space indentation, and `camelCase` for values/functions, `PascalCase` for types.
- Keep the pure core free of `IO`; isolate side effects in `Runner` or adapter modules.
- Wrap Hasktorch tensors in domain-specific newtypes (e.g., `NeedVec`) and expose helper functions instead of raw tensor ops.
- Document every module and public function with Haddock comments (`-- |`), describing intent, inputs, and return values.
- Prefer pattern matching + total functions; add concise comments only where intent is non-obvious.
- Documentation comments: prefer GHC-style `Note [Stable Id]` blocks (with unique, stable identifiers) and point from source comments to related requirements, specs, or design docs.
- Escalate sandbox or permission errors to the project owner immediately; do not attempt environment workarounds without approval.

## Testing Guidelines
- Use `hspec` for unit and property tests; integrate `hedgehog` for stochastic behaviors (e.g., `validTime`).
- Place test modules under `test/Core/...` mirroring source paths, suffixed with `Spec`.
- Add golden scenarios for small agent populations to verify deterministic dynamics and pricing.
- Ensure every PR keeps `cabal test` green; add regression tests alongside bug fixes.

## Commit & Pull Request Guidelines
- Write commits in imperative mood (`Add tensor helper for NeedVec`), scoped narrowly per change.
- Reference relevant docs updates when altering design or behavior (`docs/Specification.md`, `docs/ImplementationPlan.md`).
- PRs should include: summary, testing evidence (`cabal test` output), linked issues, and any new configuration steps. Attach screenshots only if UI artifacts are added (rare here).

## Tooling & Remote Resources
- Web search is enabled in the development environment; consult upstream Hasktorch docs or economics references before making major architectural decisions.
- Capture any key findings in `docs/Specification.md` or inline module Haddocks when they influence implementation notes.
