{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE LambdaCase #-}

module Main (main) where

import Core.Tensor qualified as Tensor
import Core.Agent qualified as Agent
import Core.Config qualified as Config
import Core.Dynamics qualified as Dynamics
import Core.Matrix qualified as Matrix
import Core.Simulation qualified as Simulation
import Core.World qualified as World
import Core.Types
import System.Exit (exitFailure)

main :: IO ()
main = do
  results <- sequence tests
  if and results
    then putStrLn "All tests passed"
    else exitFailure

tests :: [IO Bool]
tests =
  [ expectRight "needVec accepts non-negative entries" (needVec [0.1, 0.5])
  , expectLeft  "needVec rejects negatives" (needVec [0.1, -0.2])
  , expectRight "spendVec accepts positive" (spendVec [1, 2, 3])
  , validateMatrixVector
  , validateDefaultEnv
  , validateEvolveNeeds
  , validateWorldBuilder
  , validateStepAgent
  , validateStepWorld
  , validateTimeBudget
  , validateMoneyNext
  , expectLeft "moneyNext dimension mismatch" moneyMismatch
  , expectLeft "world price dimension mismatch" worldMismatch
  ]

moneyMismatch :: Either DomainError Money
moneyMismatch =
  let ent = Tensor.fromList []
  in do
    prices <- priceVec [5, 2]
    spends <- spendVec [1]
    alloc  <- timeAlloc ent 1 0 0 0
    moneyNext 10 prices 0 alloc spends

validateTimeBudget :: IO Bool
validateTimeBudget = do
  let ent = Tensor.fromList [0.5, 0.5]
  case timeAlloc ent 1 1 1 1 of
    Left _ -> report False "timeAlloc should accept non-negative hours"
    Right alloc ->
      let total = timeAllocTotal alloc
      in case validTimeAlloc 5 alloc of
           Left _ -> report False "validTimeAlloc should accept within budget"
           Right _ ->
             if abs (total - 5) < 1e-6
               then report True "timeAllocTotal sums correctly"
               else report False "timeAllocTotal sums incorrectly"

validateMoneyNext :: IO Bool
validateMoneyNext = do
  let ent = Tensor.fromList []
  case (priceVec [5, 2], spendVec [1, 0.5], timeAlloc ent 1 0 0 0) of
    (Right prices, Right spends, Right alloc) ->
      case moneyNext 10 prices 100 alloc spends of
        Right balance ->
          let expected = 100 + 10 * 1 - (5 * 1 + 2 * 0.5)
          in if abs (balance - expected) < 1e-6
               then report True "moneyNext computes balance"
               else report False "moneyNext incorrect"
        Left err -> report False ("moneyNext failed with " <> show err)
    _ -> report False "Setup for moneyNext test failed"

expectRight :: String -> Either DomainError a -> IO Bool
expectRight label = \case
  Right _ -> report True label
  Left  e -> report False (label <> " but failed with " <> show e)

expectLeft :: String -> Either DomainError a -> IO Bool
expectLeft label = \case
  Left _  -> report True label
  Right _ -> report False (label <> " but succeeded")

report :: Bool -> String -> IO Bool
report success label = do
  putStrLn $ (if success then "[PASS] " else "[FAIL] ") <> label
  pure success

approxList :: [Float] -> [Float] -> Bool
approxList xs ys = length xs == length ys && and (zipWith approx xs ys)
  where
    approx a b = abs (a - b) < 1e-5
validateMatrixVector :: IO Bool
validateMatrixVector =
  case (Matrix.fromLists [[1, 0], [0, 1]], needVec [2, 3]) of
    (Right m, Right vec) ->
      case Matrix.matrixVector m (unNeedVec vec) of
        Right res ->
          if Tensor.toList res == [2, 3]
            then report True "matrixVector (identity) leaves vector unchanged"
            else report False "matrixVector incorrect"
        Left err -> report False ("matrixVector failed with " <> show err)
    _ -> report False "matrixVector setup failed"

worldMismatch :: Either DomainError World.World
worldMismatch = do
  env <- Config.defaultEnv
  prices <- priceVec [1, 2]
  World.mkWorld env prices 10

validateDefaultEnv :: IO Bool
validateDefaultEnv =
  case Config.defaultEnv of
    Right env -> do
      let rowsOk = Matrix.rows (Config.timeEffects env) == Config.needsDimension
          colsOk = Matrix.cols (Config.reboundEffects env) == Config.entDimension
      if rowsOk && colsOk
        then report True "defaultEnv dimensions"
        else report False "defaultEnv dimensions wrong"
    Left err -> report False ("defaultEnv failed with " <> show err)

validateEvolveNeeds :: IO Bool
validateEvolveNeeds =
  let ent = Tensor.fromList [0.4, 0.2]
  in case ( Config.defaultEnv
          , needVec [1, 1, 1]
          , spendVec [0.5, 0.25, 0]
          , timeAlloc ent 1 0.5 0.2 0.3
          ) of
       (Right env, Right baseNeeds, Right spends, Right alloc) -> do
         let manual = do
               timed <- Dynamics.applyTimeTransition env baseNeeds
               spendDelta <- Dynamics.applySpendDelta env spends
               reboundDelta <- Dynamics.applyRebound env alloc
               pure (needVecZipWith (+) (needVecZipWith (+) timed spendDelta) reboundDelta)
         case (manual, Dynamics.evolveNeeds env baseNeeds spends alloc) of
           (Right expected, Right actual) ->
             if approxList (needVecToList expected) (needVecToList actual)
               then report True "evolveNeeds matches composed transitions"
               else report False "evolveNeeds mismatch"
           (Left err, _) -> report False ("manual computation failed " <> show err)
           (_, Left err) -> report False ("evolveNeeds failed " <> show err)
       _ -> report False "evolveNeeds setup failed"

validateWorldBuilder :: IO Bool
validateWorldBuilder =
  case (Config.defaultEnv, priceVec [1, 1, 1]) of
    (Right env, Right prices) ->
      case World.mkWorld env prices 12 of
        Right world ->
         if World.goodsCount world == Config.needsDimension
             then report True "mkWorld accepts matching prices"
             else report False "goodsCount mismatch"
        Left err -> report False ("mkWorld failed " <> show err)
    _ -> report False "mkWorld setup failed"

validateStepAgent :: IO Bool
validateStepAgent =
  let ent = Tensor.fromList (replicate Config.entDimension 0)
  in case ( Config.defaultEnv
          , priceVec (replicate Config.needsDimension 1)
          , needVec (replicate Config.needsDimension 1)
          , needVec (replicate Config.needsDimension 1)
          , zeroSpendPlan Config.needsDimension
          , timeAlloc ent 1 0 0 0
          ) of
       (Right env, Right prices, Right needs0, Right weights, Right baseSpend, Right alloc) -> do
         let params0 = AgentParams { setPoint = needs0
                                   , needWeight = weights
                                   , discount = 0.9
                                   , hedonicW = 0.1
                                   , gapPenalty = 0.1
                                   , timeBudget = 5
                                   }
             agent0 = Agent { agentId = 1, needs = needs0, money = 10, params = params0 }
         case ( World.mkWorld env prices 1
              , spendVec (replicate Config.needsDimension 0.1)
              ) of
           (Right world, Right spendVec0) ->
             let decision = Decision { timeAllocation = alloc, spendPlan = baseSpend { goodsSpend = spendVec0 } }
                 expectedMoney = 10 + laborHours alloc * 1 - sum (spendVecToList spendVec0)
             in case Agent.stepAgent world agent0 decision of
                  Right outcome ->
                    if abs (money (Agent.updatedAgent outcome) - expectedMoney) < 1e-6
                       then report True "stepAgent updates agent"
                       else report False "stepAgent money mismatch"
                  Left err -> report False ("stepAgent failed " <> show err)
           _ -> report False "World or spend setup failed"
       _ -> report False "stepAgent preconditions failed"

validateStepWorld :: IO Bool
validateStepWorld =
  let ent = Tensor.fromList (replicate Config.entDimension 0)
  in case ( Config.defaultEnv
          , priceVec (replicate Config.needsDimension 1)
          , needVec (replicate Config.needsDimension 1)
          , zeroSpendPlan Config.needsDimension
          , timeAlloc ent 1 0 0 0
          ) of
       (Right env, Right prices, Right needs0, Right baseSpend, Right alloc) -> do
         let params0 = AgentParams { setPoint = needs0
                                   , needWeight = needs0
                                   , discount = 0.9
                                   , hedonicW = 0.1
                                   , gapPenalty = 0.1
                                   , timeBudget = 5
                                   }
             agent0 = Agent { agentId = 1, needs = needs0, money = 10, params = params0 }
         case ( World.mkWorld env prices 1
              , spendVec (replicate Config.needsDimension 0.1)
              ) of
           (Right world, Right spendVec0) ->
             let decision = Decision { timeAllocation = alloc, spendPlan = baseSpend { goodsSpend = spendVec0 } }
             in case Simulation.stepWorld world [agent0] [decision] of
                  Right batch ->
                    if length (Simulation.agentOutcomes batch) == 1
                       then report True "stepWorld processes agents"
                       else report False "stepWorld outcome size mismatch"
                  Left err -> report False ("stepWorld failed " <> show err)
           _ -> report False "World or spend setup failed"
       _ -> report False "stepWorld preconditions failed"
