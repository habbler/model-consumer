{-# LANGUAGE DerivingStrategies #-}
-- | Shared domain error type.
module Core.Error
  ( DomainError(..)
  ) where

-- | Validation failures surfaced to callers.
data DomainError
  = NegativeValue !String !Float
  | TimeBudgetExceeded !Float !Float
  | DimensionMismatch !String !Int !Int
  deriving stock (Eq, Show)
