{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE NamedFieldPuns #-}
-- | World state representation and helpers.
module Core.World
  ( World(..)
  , mkWorld
  , adjustPrices
  , goodsCount
  ) where

import Core.Config qualified as Config
import Core.Types

data World = World
  { envMatrices :: !Config.EnvMatrices
  , priceVector :: !PriceVec
  , wageRate    :: !Price
  }
  deriving stock (Eq, Show)

mkWorld :: Config.EnvMatrices -> PriceVec -> Price -> Either DomainError World
mkWorld env prices wage
  | priceVecLength prices /= Config.needsDimension =
      Left (DimensionMismatch "Prices" Config.needsDimension (priceVecLength prices))
  | wage < 0 = Left (NegativeValue "wage" wage)
  | otherwise = Right World
      { envMatrices = env
      , priceVector = prices
      , wageRate = wage
      }

adjustPrices :: (PriceVec -> Either DomainError PriceVec) -> World -> Either DomainError World
adjustPrices f world@World{priceVector} = do
  prices' <- f priceVector
  mkWorld (envMatrices world) prices' (wageRate world)

goodsCount :: World -> Int
goodsCount World{priceVector} = priceVecLength priceVector
