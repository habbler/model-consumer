{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
-- | Note [Tensor Abstraction]
-- This module provides a lightweight tensor abstraction backed by
-- simple lists. The intent is to keep the API surface close to
-- what we expect from Hasktorch while remaining easy to test locally and
-- free of heavy dependencies in early development.
module Core.Tensor
  ( Tensor
  , fromList
  , toList
  , replicate
  , zipWith
  , map
  , foldl'
  , sum
  , dot
  , length
  , all
  , any
  ) where

import Prelude hiding (all, any, foldl', length, map, replicate, sum, zipWith)
import qualified Data.List as List

-- | Note [Tensor representation]
-- A thin wrapper around lists so we can swap in real Hasktorch tensors
-- later without disturbing the domain API.
newtype Tensor = Tensor { unTensor :: [Float] }
  deriving stock (Eq, Show)
  deriving newtype (Semigroup, Monoid)

-- | Construct a tensor from a list.
fromList :: [Float] -> Tensor
fromList = Tensor

-- | Convert a tensor back to a list.
toList :: Tensor -> [Float]
toList = unTensor

-- | Replicate a scalar value @n@ times.
replicate :: Int -> Float -> Tensor
replicate n x = Tensor (List.replicate n x)

-- | Map a function across all elements.
map :: (Float -> Float) -> Tensor -> Tensor
map f = Tensor . List.map f . unTensor

-- | Zip two tensors with a binary function. Asserts equal length.
zipWith :: (Float -> Float -> Float) -> Tensor -> Tensor -> Tensor
zipWith f (Tensor xs) (Tensor ys) =
  if List.length xs /= List.length ys
    then error "zipWith: tensors must have equal length"
    else Tensor (List.zipWith f xs ys)

-- | Left fold over tensor elements.
foldl' :: (a -> Float -> a) -> a -> Tensor -> a
foldl' f z (Tensor xs) = List.foldl' f z xs

-- | Sum tensor elements.
sum :: Tensor -> Float
sum = foldl' (+) 0

-- | Dot product of two tensors (asserts equal length).
dot :: Tensor -> Tensor -> Float
dot xs ys = sum (zipWith (*) xs ys)

-- | Length of a tensor.
length :: Tensor -> Int
length (Tensor xs) = List.length xs

-- | Are all elements satisfying predicate?
all :: (Float -> Bool) -> Tensor -> Bool
all p = List.all p . unTensor

-- | Does any element satisfy predicate?
any :: (Float -> Bool) -> Tensor -> Bool
any p = List.any p . unTensor
