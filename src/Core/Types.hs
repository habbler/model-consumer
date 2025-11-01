{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RecordWildCards #-}
-- | Note [Core types]
-- Provides foundational domain types and helpers that keep the pure
-- world state independent of IO concerns. Tensor-backed newtypes keep
-- the interface close to what the specification demands while remaining
-- easy to test.
module Core.Types
  ( Need
  , Money
  , TimeHours
  , Prob
  , Price
  , NeedVec(..)
  , SpendVec(..)
  , PriceVec(..)
  , TimeAlloc(..)
  , AgentParams(..)
  , Agent(..)
  , SpendPlan(..)
  , Decision(..)
  , needVec
  , spendVec
  , priceVec
  , timeAlloc
  , timeAllocTotal
  , validTimeAlloc
  , spendTotal
  , moneyNext
  , validateDecision
  , zeroSpendPlan
  , needVecMap
  , needVecZipWith
  , needVecToList
  , spendVecToList
  , priceVecToList
  , priceVecLength
  , module Core.Error
  ) where

import Core.Config qualified as Config
import Core.Error (DomainError(..))
import Core.Tensor (Tensor)
import qualified Core.Tensor as Tensor

type Need = Float
type Money = Float
type TimeHours = Float
type Prob = Float
type Price = Float

-- | Wrapper for need vectors.
newtype NeedVec = NeedVec { unNeedVec :: Tensor }
  deriving stock (Eq, Show)

-- | Wrapper for spend vectors (per-good spend amounts).
newtype SpendVec = SpendVec { unSpendVec :: Tensor }
  deriving stock (Eq, Show)

-- | Wrapper for price vectors.
newtype PriceVec = PriceVec { unPriceVec :: Tensor }
  deriving stock (Eq, Show)

-- | Allocation of time across activities (entertainment tensor is arity-k).
data TimeAlloc = TimeAlloc
  { laborHours  :: !TimeHours
  , foodHours   :: !TimeHours
  , socialHours :: !TimeHours
  , restHours   :: !TimeHours
  , entertainment :: !Tensor
  }
  deriving stock (Eq, Show)

data AgentParams = AgentParams
  { setPoint   :: !NeedVec
  , needWeight :: !NeedVec
  , discount   :: !Float
  , hedonicW   :: !Float
  , gapPenalty :: !Float
  , timeBudget :: !TimeHours
  }
  deriving stock (Eq, Show)

data Agent = Agent
  { agentId :: !Int
  , needs   :: !NeedVec
  , money   :: !Money
  , params  :: !AgentParams
  }
  deriving stock (Eq, Show)

data SpendPlan = SpendPlan
  { goodsSpend :: !SpendVec
  }
  deriving stock (Eq, Show)

data Decision = Decision
  { timeAllocation :: !TimeAlloc
  , spendPlan      :: !SpendPlan
  }
  deriving stock (Eq, Show)

-- | Validation failures surfaced to callers.
-- | Build a need vector from a list of needs.
needVec :: [Float] -> Either DomainError NeedVec
needVec xs
  | any (< 0) xs = Left (NegativeValue "NeedVec" (minimum xs))
  | null xs = Left (DimensionMismatch "NeedVec" 0 0)
  | otherwise = Right (NeedVec (Tensor.fromList xs))

-- | Build a spend vector.
spendVec :: [Float] -> Either DomainError SpendVec
spendVec xs
  | any (< 0) xs = Left (NegativeValue "SpendVec" (minimum xs))
  | null xs = Left (DimensionMismatch "SpendVec" 0 0)
  | otherwise = Right (SpendVec (Tensor.fromList xs))

-- | Build a price vector.
priceVec :: [Float] -> Either DomainError PriceVec
priceVec xs
  | any (< 0) xs = Left (NegativeValue "PriceVec" (minimum xs))
  | null xs = Left (DimensionMismatch "PriceVec" 0 0)
  | otherwise = Right (PriceVec (Tensor.fromList xs))

-- | Smart constructor for @TimeAlloc@ that ensures non-negative durations.
timeAlloc
  :: Tensor -- ^ entertainment hours
  -> TimeHours -> TimeHours -> TimeHours -> TimeHours
  -> Either DomainError TimeAlloc
timeAlloc entertainment laborHours foodHours socialHours restHours
  | laborHours < 0 = Left (NegativeValue "laborHours" laborHours)
  | foodHours < 0 = Left (NegativeValue "foodHours" foodHours)
  | socialHours < 0 = Left (NegativeValue "socialHours" socialHours)
  | restHours < 0 = Left (NegativeValue "restHours" restHours)
  | Tensor.any (< 0) entertainment =
      Left (NegativeValue "entertainment" (minimum (Tensor.toList entertainment)))
  | otherwise =
      Right TimeAlloc
        { laborHours
        , foodHours
        , socialHours
        , restHours
        , entertainment
        }

-- | Total hours allocated including entertainment activities.
timeAllocTotal :: TimeAlloc -> TimeHours
timeAllocTotal TimeAlloc{laborHours, foodHours, socialHours, restHours, entertainment} =
  laborHours + foodHours + socialHours + restHours + Tensor.sum entertainment

-- | Check whether a time allocation fits within the agent's budget.
validTimeAlloc :: TimeHours -> TimeAlloc -> Either DomainError TimeAlloc
validTimeAlloc budget alloc =
  let total = timeAllocTotal alloc
  in if total <= budget
       then Right alloc
       else Left (TimeBudgetExceeded budget total)

-- | Total spend for a basket of goods at current prices.
spendTotal :: PriceVec -> SpendVec -> Either DomainError Money
spendTotal (PriceVec prices) (SpendVec spends)
  | priceLen /= spendLen =
      Left (DimensionMismatch "Spend/Price" spendLen priceLen)
  | otherwise = Right (Tensor.dot prices spends)
  where
    priceLen = Tensor.length prices
    spendLen = Tensor.length spends

-- | Compute next money balance after labor income and consumption spend.
moneyNext
  :: Price              -- ^ wage per hour
  -> PriceVec           -- ^ price vector for goods
  -> Money              -- ^ initial balance
  -> TimeAlloc
  -> SpendVec
  -> Either DomainError Money
moneyNext wage prices balance alloc spends = do
  totalSpend <- spendTotal prices spends
  pure (balance + wage * laborHours alloc - totalSpend)

validateDecision :: AgentParams -> Decision -> Either DomainError Decision
validateDecision AgentParams{timeBudget} decision@Decision{timeAllocation}
  | Tensor.length (entertainment timeAllocation) /= Config.entDimension =
      Left (DimensionMismatch "entertainment" Config.entDimension (Tensor.length (entertainment timeAllocation)))
  | spendVectorLen /= Config.needsDimension =
      Left (DimensionMismatch "spend plan" Config.needsDimension spendVectorLen)
  | otherwise = do
      _ <- validTimeAlloc timeBudget timeAllocation
      pure decision
  where
    spendVectorLen = Tensor.length (unSpendVec (goodsSpend (spendPlan decision)))

zeroSpendPlan :: Int -> Either DomainError SpendPlan
zeroSpendPlan n
  | n <= 0 = Left (DimensionMismatch "SpendPlan" n n)
  | otherwise = do
      let zeros = SpendVec (Tensor.replicate n 0)
      pure SpendPlan { goodsSpend = zeros }

needVecMap :: (Float -> Float) -> NeedVec -> NeedVec
needVecMap f (NeedVec t) = NeedVec (Tensor.map f t)

needVecZipWith :: (Float -> Float -> Float) -> NeedVec -> NeedVec -> NeedVec
needVecZipWith f (NeedVec a) (NeedVec b) = NeedVec (Tensor.zipWith f a b)

needVecToList :: NeedVec -> [Float]
needVecToList (NeedVec t) = Tensor.toList t

spendVecToList :: SpendVec -> [Float]
spendVecToList (SpendVec t) = Tensor.toList t

priceVecToList :: PriceVec -> [Float]
priceVecToList (PriceVec t) = Tensor.toList t

priceVecLength :: PriceVec -> Int
priceVecLength (PriceVec t) = Tensor.length t
