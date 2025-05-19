// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// TODO(kostiantynl): add collectives
//     stablehlo::AllReduceOp, stablehlo::AllGatherOp,
//     stablehlo::AllToAllOp, stablehlo::CollectivePermuteOp
//     CollectiveBroadcast
// CHECK-NOT: reshard
