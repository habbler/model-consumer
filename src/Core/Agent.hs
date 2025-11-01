{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE NamedFieldPuns #-}
module Core.Agent
  ( StepOutcome(..)
  , stepAgent
  ) where

import Core.Dynamics qualified as Dynamics
import Core.Types
import Core.World (World(..))

data StepOutcome = StepOutcome
  { updatedAgent :: !Agent
  , rewardEarned :: !Float
  }
  deriving stock (Eq, Show)

stepAgent :: World -> Agent -> Decision -> Either DomainError StepOutcome
stepAgent world agent decision = do
  _ <- validateDecision (params agent) decision
  let env = envMatrices world
      spends = goodsSpend (spendPlan decision)
      alloc  = timeAllocation decision
  newNeeds <- Dynamics.evolveNeeds env (needs agent) spends alloc
  newMoney <- moneyNext (wageRate world) (priceVector world) (money agent) alloc spends
  let updated = agent { needs = newNeeds, money = newMoney }
  pure StepOutcome { updatedAgent = updated, rewardEarned = 0 }
