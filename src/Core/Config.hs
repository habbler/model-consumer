{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE NamedFieldPuns #-}
-- | Static configuration and helpers for domain matrices and constants.
module Core.Config
  ( EnvMatrices(..)
  , defaultEnv
  , needsDimension
  , entDimension
  , ensureNeedDim
  , ensureEntDim
  ) where

import Core.Error (DomainError(..))
import Core.Matrix (Matrix)
import qualified Core.Matrix as Matrix

-- | Bundle of matrices governing state transitions.
data EnvMatrices = EnvMatrices
  { timeEffects   :: !Matrix
  , spendEffects  :: !Matrix
  , reboundEffects :: !Matrix
  , substitution  :: !Matrix
  }
  deriving stock (Eq, Show)

needsDimension :: Int
needsDimension = 3

entDimension :: Int
entDimension = 2

ensureNeedDim :: Matrix -> Either DomainError Matrix
ensureNeedDim m
  | Matrix.rows m /= needsDimension = Left (DimensionMismatch "Need rows" needsDimension (Matrix.rows m))
  | Matrix.cols m /= needsDimension = Left (DimensionMismatch "Need cols" needsDimension (Matrix.cols m))
  | otherwise = Right m

ensureEntDim :: Matrix -> Either DomainError Matrix
ensureEntDim m
  | Matrix.rows m /= needsDimension = Left (DimensionMismatch "Need rows" needsDimension (Matrix.rows m))
  | Matrix.cols m /= entDimension = Left (DimensionMismatch "Ent cols" entDimension (Matrix.cols m))
  | otherwise = Right m

defaultEnv :: Either DomainError EnvMatrices
defaultEnv = do
  t <- ensureNeedDim =<< Matrix.fromLists
         [ [0.90, -0.05, 0.00]
         , [-0.05, 0.92, -0.03]
         , [0.00, -0.03, 0.95]
         ]
  s <- ensureNeedDim =<< Matrix.fromLists
         [ [0.02, 0.01, 0.00]
         , [0.01, 0.03, 0.01]
         , [0.00, 0.01, 0.02]
         ]
  r <- ensureEntDim =<< Matrix.fromLists
         [ [0.10, 0.05]
         , [0.08, 0.04]
         , [0.05, 0.02]
         ]
  sub <- ensureEntDim =<< Matrix.fromLists
         [ [0.3, 0.2]
         , [0.1, 0.3]
         , [0.2, 0.1]
         ]
  pure EnvMatrices
        { timeEffects = t
        , spendEffects = s
        , reboundEffects = r
        , substitution = sub
        }
