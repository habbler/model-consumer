{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE NamedFieldPuns #-}
-- | Note [Matrix abstraction]
-- Provides a minimal row-major matrix representation with helpers to
-- validate dimensions and perform matrix-vector products. This keeps the
-- domain core close to future Hasktorch usage while remaining lightweight
-- for early development and testing.
module Core.Matrix
  ( Matrix(..)
  , fromLists
  , toLists
  , matrixVector
  , identity
  ) where

import Core.Error (DomainError(..))
import Core.Tensor (Tensor)
import qualified Core.Tensor as Tensor

data Matrix = Matrix
  { rows :: !Int
  , cols :: !Int
  , cells :: ![Float]  -- ^ row-major storage
  }
  deriving stock (Eq, Show)

fromLists :: [[Float]] -> Either DomainError Matrix
fromLists [] = Left (DimensionMismatch "Matrix" 0 0)
fromLists xss@(row0:_) =
  if any null xss
     then Left (DimensionMismatch "Matrix" 0 0)
     else
       let expectedCols = length row0
       in if any ((/= expectedCols) . length) xss
            then Left (DimensionMismatch "Matrix" expectedCols (maximum (map length xss)))
            else Right Matrix
                   { rows = length xss
                   , cols = expectedCols
                   , cells = concat xss
                   }

toLists :: Matrix -> [[Float]]
toLists Matrix{rows, cols, cells} = chunk cols rows cells

matrixVector :: Matrix -> Tensor -> Either DomainError Tensor
matrixVector m@Matrix{cols} vec
  | Tensor.length vec /= cols =
      Left (DimensionMismatch "Matrix-Vector" cols (Tensor.length vec))
  | otherwise =
      let rowVecs = map Tensor.fromList (toLists m)
          result = map (`Tensor.dot` vec) rowVecs
      in Right (Tensor.fromList result)

identity :: Int -> Either DomainError Matrix
identity n
  | n <= 0 = Left (DimensionMismatch "Identity" n n)
  | otherwise =
      fromLists [ [ if i == j then 1 else 0 | j <- [1..n] ] | i <- [1..n] ]

chunk :: Int -> Int -> [Float] -> [[Float]]
chunk _ 0 _ = []
chunk w r xs =
  let (h, t) = splitAt w xs
  in h : chunk w (r - 1) t
