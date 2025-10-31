**Two situations & better questions**

1. you want an implementable plan, not another brainstorm
   → better question: “Can you give me a crisp spec + architecture + design—with types, module layout, interfaces, and pseudocode—for a loop-agnostic multi-agent economy with homeostatic needs and optional Hasktorch policies?”

2. you’ll need to swap runners (fold, streaming, concurrent) and maybe swap learning backends
   → better question: “How do we keep a pure domain core and define minimal ports so different simulation loops and policy learners (tabular/Hasktorch) drop in without rewriting?”

# Spec, Architecture, & Design (with pseudocode)

## 0) Scope & goals

* **Goal:** Simulate a population of agents who allocate **time** and **money** to satisfy **homeostatic needs** (e.g., energy, hunger, belonging, safety, competence). Agents may also consume **entertainment**, which partially substitutes some needs and can produce rebound/illusion effects. Goods and labor **markets** set prices/wages via simple clearing.
* **Loop-agnostic:** The **domain core is pure** (no IO, no scheduler). Any outer loop (fold, FRP, DES, STM/async) can drive ticks.
* **Pluggable policies:** Policies can be **tabular/heuristic** or **learned** (Hasktorch). The world logic doesn’t depend on the learner.
* **Scale target:** Initially 1k–10k agents on CPU.
* **Numeric backbone:** All vectors/matrices use **Hasktorch** `Tensor`s (Float/DType Float), wrapped in small newtypes where helpful for clarity; helper functions keep tensor operations pure.

---

## 1) Core concepts & data model

### 1.1 Types (Haskell-ish)

```haskell
-- Core scalars and vectors (kept in Float to align with Hasktorch tensor dtype)
type Need      = Float
type Money     = Float
type TimeHours = Float
type Prob      = Float
type Price     = Float

-- Fixed small vector of needs: [energy, hunger, belonging, safety, competence, ...]
newtype NeedVec = NeedVec Tensor          -- 1D tensor length nNeeds, DType Float

-- Per-agent immutable parameters
data AgentParams = AgentParams
  { setPoint   :: !NeedVec             -- h*
  , needWeight :: !NeedVec             -- W (importance/tolerance per need)
  , discount   :: !Float               -- β
  , hedonicW   :: !Float               -- ω_hed
  , gapPenalty :: !Float               -- λ_gap
  , timeBudget :: !TimeHours           -- T per tick
  }

-- Agent dynamic state (pure)
data Agent = Agent
  { needs   :: !NeedVec                -- h_t
  , beliefs :: !(Maybe NeedVec)        -- \hat{h}_t (optional; Nothing = no illusions model)
  , money   :: !Money
  , params  :: !AgentParams
  , idA     :: !AgentId
  }

-- Entertainment type index K and substitution/rebound profiles
newtype EntType = EntType Int

-- World state
data World = World
  { prices   :: !(HashMap Good Price)      -- goods incl. entertainment types
  , wage     :: !Price                      -- wage per hour
  , network  :: !(Graph AgentId)            -- social graph
  , shocks   :: !WorldShocks
  , goods    :: !(Vector Good)              -- catalog
  }

-- Decisions (per agent per tick)
data Decision = Decision
  { timeAlloc :: TimeAlloc      -- split of T across {labor, eat, social, ent[k], rest}
  , spends    :: SpendVec       -- amounts per good (money)
  } deriving (Eq, Show)

-- Time allocation is small fixed tuple
data TimeAlloc = TimeAlloc
  { labor   :: !TimeHours
  , eat     :: !TimeHours
  , social  :: !TimeHours
  , rest    :: !TimeHours
  , ent     :: !Tensor              -- 1D tensor, hours per entertainment type k
  }

-- Market orders (derived from decisions)
data Order = Order { good :: Good, qty :: Float, buyer :: AgentId }

-- Observables provided to policies (no IO)
data Obs = Obs
  { me       :: !AgentSummary        -- minimal view of self
  , market   :: !MarketSummary       -- prices, wage, aggregates
  , social   :: !SocialSummary       -- egocentric network stats
  }

-- Policy interface (pure; learning can be offloaded elsewhere)
type Policy = Obs -> Agent -> Decision
```

### 1.2 Matrices & functions (domain parameters)

* **A (time effects):** how time in activities changes *true needs*.
* **B (spend effects):** how spending on goods/services changes *true needs*.
* **S (substitution):** entertainment illusion matrix that contracts **perceived** gaps.
* **R (rebound):** delayed costs to *true needs* after entertainment overuse.
* **Drive function:** convex loss over deviation from set-points.

```haskell
-- Drive: D(h) = sum_i w_i * loss(h_i - h*_i)
drive :: NeedVec -> AgentParams -> Float

-- State transition for needs (true)
applyTimeEffects  :: TimeAlloc -> NeedVec -> NeedVec
applySpendEffects :: SpendVec  -> NeedVec -> NeedVec
applyRebound      :: Rebound -> TimeAlloc -> NeedVec -> NeedVec

-- Perception (if beliefs modeled)
illusoryBoost :: TimeAlloc -> NeedVec -> NeedVec  -- uses S and hedonic saturation to move perceived gaps
updateBeliefs :: NeedVec -> NeedVec -> NeedVec    -- e.g., simple filter toward observations
```

---

## 2) Domain rules (pure)

### 2.1 Reward function (homeostatic RL with hedonic & gap terms)

[
r_t ;=; \underbrace{D(h_t)-D(h_{t+1})}*{\text{drive reduction}}
;+; \omega*{\text{hed}}, u_{\text{hed}}(\tau^{\text{ent}})
;-; \lambda_{\text{gap}}, |h_{t+1}-\hat{h}_{t+1}|_1
;-; \text{costs}(\tau, x)
]

```haskell
reward :: Agent -> Agent -> AgentParams -> Float
-- reward oldAgent newAgent params
```

### 2.2 Budget & feasibility

```haskell
-- Money update
moneyNext :: World -> Agent -> Decision -> Money
moneyNext World{wage,prices} Agent{money} Decision{timeAlloc,spends} =
  let income = wage * labor timeAlloc
      out    = dotSpend prices spends
  in money + income - out

-- Time constraint must hold (within epsilon)
validTime :: AgentParams -> TimeAlloc -> Bool
```

### 2.3 Pure agent step

```haskell
-- Core step: apply decision to agent inside current world; returns new agent and immediate reward
stepAgent :: Env -> World -> Obs -> Agent -> Decision -> (Agent, Float)
stepAgent Env{A,B,S,R,hedonicFcn,gapNorm} world obs a0 d =
  let h0       = needs a0
      h1_true  = applyRebound R (timeAlloc d) $ applySpendEffects B spends $ applyTimeEffects A (timeAlloc d) h0

      -- Perception (optional)
      h1_obs   = case beliefs a0 of
                   Nothing   -> h1_true
                   Just hat0 -> let hat1 = updateBeliefs h1_true (illusoryBoost (timeAlloc d) hat0)
                                in hat1

      a1 = a0 { needs   = h1_true
              , beliefs = fmap (const h1_obs) (beliefs a0)
              , money   = moneyNext world a0 d
              }

      r  = reward a0 a1 (params a0) - actionCosts d
  in (a1, r)
```

*(All helper functions are pure and small; `Env` bundles A,B,S,R and penalty weights.)*

---

## 3) Policies (interchangeable)

### 3.1 Pure heuristic/tabular policy

```haskell
type QTable = HashMap StateKey (HashMap Action Float)

discretize :: Agent -> World -> StateKey
decodeAction :: Int -> Decision

policyQ :: QTable -> Policy
policyQ q obs ag =
  let s   = discretize ag (reifyWorld obs)
      act = argmaxIndex (q ! s)
  in decodeAction act
```

### 3.2 Hasktorch policy (thin impure adapter at the edge)

```haskell
-- Pure featurizer
featurize :: Obs -> Agent -> Tensor   -- 1D tensor features compatible with Hasktorch models

-- Impure boundary (called by the runner, not by core logic)
decideWithMLP :: MLP -> (Obs -> Agent -> Tensor) -> Obs -> Agent -> IO Decision
```

**Design note:** The **core** only needs `Policy` (pure). The **runner** may choose to obtain a `Decision` by calling a Hasktorch model in `IO` and then hand that `Decision` to `stepAgent`.

---

## 4) Markets (pure)

### 4.1 Order construction and clearing

```haskell
ordersFrom :: World -> Agent -> Decision -> [Order]
-- translate spends/time choices into demand quantities

tatonnement :: World -> [Order] -> World
tatonnement w os =
  let excess = aggregateExcess os (supply w)
      prices' = adjust (prices w) excess
  in w { prices = prices' }

clearLabor :: World -> [TimeAlloc] -> World
-- adjust wage towards equality of supply/demand
```

*You can swap tatonnement with any clearing rule; world update stays pure.*

---

## 5) Simulation “runners” (loop-agnostic)

The **runner owns time**; the **core owns economics & psychology**. Provide multiple runners as examples; users can write their own.

### 5.1 Single-thread fold runner (pure decisions)

```haskell
runFold :: Int -> World -> Vector Agent -> Policy -> (World, Vector Agent)
runFold steps w0 as0 policy = iterateN steps step (w0, as0)
 where
  step (w, as) =
    let obs = mkObs w as
        -- pure decisions
        ds  = V.map (\a -> policy obs a) as
        -- apply steps
        ars = V.zipWith (\a d -> stepAgent env w obs a d) as ds
        as' = V.map fst ars
        w'  = tatonnement w (concat (V.toList (V.zipWith (ordersFrom w) as ds)))
    in (w', as')
```

### 5.2 Mixed runner with Hasktorch decisions (IO at the edge)

```haskell
runIOFold :: Int -> World -> Vector Agent -> (Obs -> Agent -> IO Decision)
          -> IO (World, Vector Agent)
runIOFold steps w0 as0 decideIO = go steps w0 as0
 where
  go 0 w as = pure (w, as)
  go n w as = do
    let obs = mkObs w as
    ds <- V.mapM (\a -> decideIO obs a) as
    -- commit a pure world step
    let ars = V.zipWith (\a d -> stepAgent env w obs a d) as ds
        as' = V.map fst ars
        w'  = tatonnement w (concat (V.toList (V.zipWith (ordersFrom w) as ds)))
    go (n-1) w' as'
```

### 5.3 Parallel runner (sketch)

* Phase 1 (parallel): compute decisions for each agent (`parMap` or `async`), possibly batching NN inference.
* Barrier: collect all `Decision`s.
* Phase 2 (pure): single world update (`tatonnement`), then pure `stepAgent` across agents (can be parallel if independent).

---

## 6) Entertainment modeling (partial substitute)

### 6.1 Substitution and rebound

* **Substitution matrix `S`**: (s_{i,k}\in[0,1]) reduces *perceived* gap in need (i) when consuming entertainment type (k).
* **Rebound vector `R_k`**: delayed positive drift in true gaps (e.g., sleep loss → energy down next tick).

Pseudocode in `illusoryBoost` & `applyRebound`:

```haskell
illusoryBoost :: TimeAlloc -> NeedVec -> NeedVec
illusoryBoost tAlloc hat0 =
  let eHours = ent tAlloc                    -- Tensor [k]
      satH   = saturation eHours             -- keep hours within hedonic bounds
      sk     = selectColumns S               -- Tensor [needs,k]
      adjust = 1 - alpha * (sk * satH)       -- broadcasted elementwise product
      newGaps = hadamard (toGaps hat0) adjust
  in fromGaps newGaps

applyRebound :: Rebound -> TimeAlloc -> NeedVec -> NeedVec
applyRebound R tAlloc h =
  let entHours = ent tAlloc                  -- Tensor [k]
      rebound  = matVec R entHours           -- R :: Tensor [needs,k], matvec via Hasktorch
  in h + rebound
```

*(Use saturating functions to avoid unrealistic full substitution.)*

---

## 7) Training (optional) with Hasktorch

* Keep training isolated in `Learning/Trainer.hs`, operating on **rollouts** produced by any runner.
* **Data flow:** Runner collects `(phi(s), action, reward, phi(s'))` per agent → batches → trainer updates policy/value nets.

```haskell
data Transition = Transition
  { sFeat :: Tensor, aIx :: Int, r :: Float, sFeat' :: Tensor, done :: Bool }

collectRollout :: Int -> (Obs -> Agent -> IO Decision) -> World -> Vector Agent -> IO [Transition]

updatePPO :: PPOConfig -> MLP -> ValueMLP -> [Transition] -> IO (MLP, ValueMLP)
```

**Batching for speed:** stack all agents’ features into a `[batch, inDim]` tensor for a single forward pass per tick.

---

## 8) Module layout (suggested)

```
src/
  Core/
    Types.hs           -- NeedVec, Agent, World, Decision, Obs, Params
    Drive.hs           -- drive, loss functions
    Dynamics.hs        -- A, B, S, R application; illusory boost; beliefs
    Step.hs            -- stepAgent, moneyNext, reward, constraints
    Market.hs          -- ordersFrom, tatonnement, clearLabor
    Observe.hs         -- mkObs, summaries
    Policy.hs          -- type Policy; Q-table helpers; heuristics
  Learning/
    PolicyMLP.hs       -- Hasktorch MLP; forward
    Decide.hs          -- decideWithMLP bridge; featurize; batching
    Trainer.hs         -- PPO/DQN training loops (optional)
  Runner/
    Fold.hs            -- runFold (pure)
    IOFold.hs          -- runIOFold (Hasktorch at the edge)
    Parallel.hs        -- parallel/multiprocessing sketch
  App/
    Main.hs            -- CLI, config load, choose runner/policy, run
```

---

## 9) Algorithmic walkthrough (per tick)

1. **Observation build:** `obs ← mkObs world agents` (pure summaries).
2. **Decision phase:**

   * If pure/tabular: `decision_i = policy obs agent_i` (pure).
   * If Hasktorch: `decision_i <- decideWithMLP net featurize obs agent_i` (IO).
3. **World update:**

   * `orders ← ordersFrom world agent_i decision_i` (pure for all i).
   * `world' ← tatonnement world orders` (pure).
4. **Agent updates:**

   * For each agent: `(agent_i', r_i) ← stepAgent env world obs agent_i decision_i` (pure).
5. **(Optional) Learning:** collect transitions, update NN in a separate step.
6. **Repeat.**

---

## 10) Pseudocode (end-to-end minimal)

```haskell
-- MAIN
let env    = loadEnv paramsJSON
let world0 = initWorld env seed
let agents0= initAgents env seed

-- choose one:
let policy   = policyQ qTable        -- pure
-- OR
let decideIO = decideWithMLP net featurize

-- run: pure fold OR IO fold
(wF, aF) <- runIOFold 365 world0 agents0 decideIO
saveStats wF aF
```

---

## 11) Calibration knobs

* **A, B:** time/spend impacts on true needs (units per hour/euro).
* **S:** substitution matrix per entertainment type (0…0.4 typical).
* **R:** rebound/costs (small negative to energy/belonging later).
* **W (needWeight):** trades between needs; heterogeneity here is crucial.
* **λ_gap:** penalizes living on “illusions.”
* **ω_hed:** hedonic short-term enjoyment weight (bounded).
* **Prices/wage dynamics:** tatonnement step size, supply caps.

---

## 12) Testing & invariants

* **Time budget:** `sum timeAlloc == T ± ε`.
* **Money:** `moneyNext ≥ 0` unless overdraft allowed.
* **Critical needs:** hard lower bounds respected unless “death/illness” modeled.
* **Market sanity:** tatonnement reduces |excess| norm each iteration.
* **Learned policy regressions:** fixed seeds produce reproducible rollouts.

---

## 13) Performance tips

* Store agent fields in **structure-of-arrays** (`Vector` per field) for cache locality.
* Batch Hasktorch inference: `[nAgents, inDim]` → one forward per tick.
* Keep features compact; prefer **discrete action heads**; only add continuous heads (e.g., Dirichlet for budget shares) if needed.
* Profile with `criterion` and `weigh` before considering heavier deps.

---

## 14) Example parameters (starter)

* Needs: `n=3` → energy, hunger, belonging; set-point all `1.0`.
* Time effects A (per hour): rest `+0.25 energy`, work `-0.20 energy`, social `+0.15 belonging`, eat `+0.10 hunger`.
* Spend effects B: food `1€ -> +0.10 hunger`, rent/security `1€ -> +0.02 safety`.
* Entertainment:

  * Video: `S(:,video) = [0.05, 0.00, 0.20]`, `R(:,video) = [ -0.05, -0.02, -0.03 ]` (applied next tick).
  * Multiplayer: `S(:,mp) = [0.00, 0.00, 0.35]`, `R(:,mp) = [ -0.08, 0.00, -0.02 ]`.
* Weights: `W = [1.0, 1.0, 1.2]`, `λ_gap = 0.6`, `ω_hed = 0.2`.
* Wage = `15 €/h`, prices: food `4 €/unit`, ent `1 €/h`.

---

## 15) When this design would be bad advice

* If you require **calendar-accurate discrete-event semantics** (queues, preemption, resource contention) and strict causality **now**, writing/maintaining your own runners may be costly and error-prone. In that case, accepting a DES framework (e.g., Aivika or Simulacron) as the loop owner would reduce complexity and bugs.

---

If you want, I can turn this into a small starter repo (skeleton modules + cabal + a toy runner and a tiny Hasktorch MLP) so you can compile and extend immediately.
