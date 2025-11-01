{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE NamedFieldPuns #-}
-- | Pure state-transition helpers built on top of matrix utilities.
module Core.Dynamics
  ( applyTimeTransition
  , applySpendDelta
  , applyRebound
  , evolveNeeds
  ) where

import Core.Config (EnvMatrices(..))
import Core.Matrix qualified as Matrix
import Core.Types

applyTimeTransition :: EnvMatrices -> NeedVec -> Either DomainError NeedVec
applyTimeTransition EnvMatrices{timeEffects} (NeedVec needs) =
  NeedVec <$> Matrix.matrixVector timeEffects needs

applySpendDelta :: EnvMatrices -> SpendVec -> Either DomainError NeedVec
applySpendDelta EnvMatrices{spendEffects} (SpendVec spends) =
  NeedVec <$> Matrix.matrixVector spendEffects spends

applyRebound :: EnvMatrices -> TimeAlloc -> Either DomainError NeedVec
applyRebound EnvMatrices{reboundEffects} TimeAlloc{entertainment} =
  NeedVec <$> Matrix.matrixVector reboundEffects entertainment

evolveNeeds
  :: EnvMatrices
  -> NeedVec
  -> SpendVec
  -> TimeAlloc
  -> Either DomainError NeedVec
evolveNeeds env needs spends alloc = do
  timed <- applyTimeTransition env needs
  spendDelta <- applySpendDelta env spends
  reboundDelta <- applyRebound env alloc
  let combined = needVecZipWith (+) timed spendDelta
  pure (needVecZipWith (+) combined reboundDelta)
