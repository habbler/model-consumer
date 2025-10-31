Implementation Plan – Loop-Agnostic Multi-Agent Economy
======================================================

1. Overview
-----------

- Goal: implement the pure domain core, pluggable policy boundary, and runner examples described in `Specification.md`, enabling simulation of agents with homeostatic needs and swap-in learning backends.
- Scope: cover domain types and dynamics, market clearing, observation and policy interfaces, runner scaffolding (pure and IO), and hooks for optional Hasktorch integration and training.
- Numeric stack: standardize on Hasktorch tensors for vectors/matrices while keeping domain APIs pure.
- Out of scope: production-grade visualization, distributed execution, or polished ML training curricula (can follow later).

2. Guiding Principles
---------------------

- Keep the core pure: world state transitions, agent updates, and markets stay referentially transparent.
- Minimize coupling: expose small, typed ports so loops, learners, and storage can vary independently.
- Validate incrementally: each phase finishes with tests or demo runners to prove correctness before layering complexity.
- Lean on Hasktorch tensors: wrap tensor math in focused helpers to preserve clarity and maintain pure interfaces.
- Parameter-driven: capture matrices, weights, and configuration data in explicit types for clarity and future calibration work.

3. Phase Roadmap
----------------

| Phase | Focus                                | Key Outputs                                                                 | Dependencies             |
|-------|---------------------------------------|------------------------------------------------------------------------------|--------------------------|
| P1    | Foundations & utilities               | Scalar/type aliases, tensor helpers, configuration stubs, fixtures          | None                     |
| P2    | Agent/state core                      | `AgentParams`, `Agent`, `NeedVec`, drive functions, needs transition logic   | P1                       |
| P3    | Markets & world updates               | `World`, goods catalog, order derivation, tatonnement, wage clearing         | P2                       |
| P4    | Observation & policy boundary         | `Obs`, summaries, `Policy` typeclass/alias, heuristic policy implementations | P2, P3                   |
| P5    | Runners & execution scaffolding       | Pure fold runner, IO runner, batching utilities, configuration CLI           | P4                       |
| P6    | Learning integration (optional tier) | Hasktorch adapters, rollout collection, trainer skeleton                     | P4, P5; requires torch   |
| P7    | Testing & validation tooling          | Property tests, golden fixtures, benchmark harnesses                         | Iterates alongside P2–P5 |

4. Detailed Tasks by Phase
--------------------------

### P1. Foundations & Utilities

- Define core numeric aliases (`Need`, `Money`, `Price`, `TimeHours`, `Prob`) and small fixed-length vectors (`NeedVec`, entertainment tensors) as Hasktorch-backed newtypes.
- Set up Hasktorch dependency wiring (Cabal flags, module imports) and prototype helper combinators for common tensor ops (zipWith, clamp, saturation functions) exposed via pure wrappers.
- Add configuration module for constant matrices (`MatrixEnv`) and load fixtures (static JSON/TOML or inline defaults) to unblock downstream modules.
- Establish shared error or validation types (e.g., `DomainError`, `FeasibilityViolation`).

### P2. Agent and State Core

- Implement `AgentParams`, `Agent`, `TimeAlloc`, `Decision`, and supporting smart constructors (enforce time budget, non-negative spends).
- Implement drive and reward functions:
  - `drive :: NeedVec -> AgentParams -> Float`
  - `reward :: Agent -> Agent -> AgentParams -> Float`
- Implement needs transition helpers:
  - `applyTimeEffects`, `applySpendEffects`, `applyRebound` (tensor pipelines)
  - `illusoryBoost`, `updateBeliefs`, `moneyNext`, `validTime`
- Implement `stepAgent :: Env -> World -> Obs -> Agent -> Decision -> (Agent, Float)` and internalize validation (return `Either DomainError` or assert via property tests).
- Provide deterministic fixtures (sample agents, params) for tests.

### P3. Markets and World Updates

- Define `Good`, `World`, `Order`, and `MarketSummary`.
- Implement `ordersFrom :: World -> Agent -> Decision -> [Order]` translating decisions to market demand.
- Implement tatonnement:
  - Aggregate excess demand.
  - Adjust prices via configurable step (damped updates, floor/ceiling).
- Implement labor market clearing (`clearLabor`) and integrate into a world-step function.
- Add `stepWorld :: Env -> World -> Vector Agent -> Vector Decision -> (World, [Order])` to centralize price updates plus optional shock handling.

### P4. Observation and Policy Boundary

- Implement observable summaries:
  - `AgentSummary`, `MarketSummary`, `SocialSummary`.
  - `mkObs :: World -> Vector Agent -> Obs`.
- Define `Policy` type alias or newtype; provide pure helper to run policy with validation.
- Implement baseline heuristic policies:
  - Greedy homeostasis fixer.
  - Budgeted proportional allocation.
- Provide discretization/encoding utilities for tabular policies.
- Ensure `Obs` exposes only permitted data (respect encapsulation for potential partial observability).

### P5. Runners and Execution Scaffolding

- Implement `runFold :: Int -> Env -> World -> Vector Agent -> Policy -> (World, Vector Agent, [Metrics])` returning metrics or logs.
- Implement `runIOFold :: Int -> Env -> World -> Vector Agent -> (Obs -> Agent -> IO Decision) -> IO (World, Vector Agent, [Metrics])`.
- Add simple CLI/app entry point allowing scenario selection, step count, and policy choice (extend `app/Main.hs`).
- Implement `Metrics` collection (reward sums, need gaps, price trends) for diagnostics.
- Provide configuration loader (maybe `yaml` or `aeson`) for scenario definitions.

### P6. Learning Integration (Optional)

- Define `Transition`/`Trajectory` types capturing rollouts.
- Implement `collectRollout` using IO runner and featurizer for features.
- Build Hasktorch adapters:
  - `featurize :: Obs -> Agent -> Tensor`
  - `decideWithMLP :: MLP -> Obs -> Agent -> IO Decision`
- Sketch trainer skeleton (e.g., PPO-style) with placeholders for optimizer steps.
- Gate all torch-dependent modules behind Cabal flags or optional packages.

### P7. Testing, Validation, Benchmarks

- Add property tests for conservation laws (money/time).
- Add golden tests for deterministic scenarios (small agent sets) verifying need evolution and prices.
- Add QuickCheck/hedgehog for `validTime`, `moneyNext`, `applyRebound`.
- Provide benchmark suite measuring per-step runtime (e.g., `criterion`).
- Add documentation tasks: literate examples in `docs` showing runner usage.

5. Cross-Cutting Concerns
-------------------------

- **Configuration**: centralize environment matrices and parameters under `Config` module with pure defaults and IO loaders.
- **Error handling**: choose between `Either DomainError` or pure exceptions; prefer explicit `Either`.
- **Performance**: profile critical tensor operations; keep tensors contiguous/typed for Hasktorch efficiency and watch for host/device copies.
- **Extensibility**: design types with future goods/needs expansions in mind (tagged newtypes, small arrays).

6. Deliverables and Acceptance
------------------------------

- Compilable modules per phase with documentation comments.
- Demonstration scripts: at least one runnable scenario showing agents stabilizing needs (pure runner) and one using IO boundary.
- Passing test suite (`cabal test`) covering drive, rewards, transitions, and market clearing.
- Optional: Hasktorch demo behind a Cabal flag, runnable if dependencies available.

7. Open Questions / Decisions to Resolve
----------------------------------------

- Preferred Hasktorch dtype/layout (Float vs Double) and whether to fix shapes via type-level helpers.
- Representation and persistence of matrices (`Env`): derive from static config vs. runtime loading.
- Level of partial observability for `Obs` (e.g., should agents see full market depth? neighbors only?).
- Policy evaluation contract: synchronous only, or allow asynchronous decision futures.
- Data logging format for runners (JSON lines vs. CSV vs. in-memory stats).

8. Next Actions
---------------

1. Finalize Hasktorch tensor wiring (dtype, device defaults) and settle `Env` structure (P1 blocker).
2. Implement P1/P2 modules with tests to anchor the rest.
3. Iterate through phases, validating with targeted tests/demos before moving forward.
