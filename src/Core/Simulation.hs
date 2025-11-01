{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE ImportQualifiedPost #-}
module Core.Simulation
  ( stepWorld
  , StepBatch(..)
  ) where

import Control.Monad (zipWithM)

import Core.Agent qualified as Agent
import Core.Types
import Core.World (World)

data StepBatch = StepBatch
  { worldAfter   :: !World
  , agentOutcomes :: ![Agent.StepOutcome]
  } deriving stock (Eq, Show)

stepWorld :: World -> [Agent] -> [Decision] -> Either DomainError StepBatch
stepWorld world agents decisions
  | length agents /= length decisions =
      Left (DimensionMismatch "stepWorld" (length agents) (length decisions))
  | otherwise = do
      outcomes <- zipWithM (Agent.stepAgent world) agents decisions
      pure StepBatch { worldAfter = world, agentOutcomes = outcomes }
