// RUN: mpmd_opt %s -mpmd-simplify-named-computations 2>&1 | FileCheck %s

// CHECK-LABEL: func @no_duplicate_operands_or_results
func.func @no_duplicate_operands_or_results(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> tensor<4x8xf32>
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: %[[NC:.*]] = mpmd.named_computation<"f"(2)> (%arg0, %arg1)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[NC]]
  %0 = mpmd.named_computation<"f"(2)> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %2 : tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @duplicate_operands
func.func @duplicate_operands(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>)
  -> tensor<4x8xf32>
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: %[[NC:.*]] = mpmd.named_computation<"f"(1)> (%arg0, %arg1, %arg2)
// CHECK-SAME:   (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg3, %arg4
// CHECK-NEXT:   %[[MUL:.*]] = stablehlo.multiply %arg5, %arg3
// CHECK-NEXT:   %[[SUB:.*]] = stablehlo.subtract %[[ADD]], %arg5
// CHECK-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[MUL]], %arg5
// CHECK-NEXT:   %[[POW:.*]] = stablehlo.power %[[SUB]], %[[DIV]]
// CHECK-NEXT:   mpmd.return %[[POW]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[NC]]
  %0 = mpmd.named_computation<"f"(1)> (%arg0, %arg1, %arg2, %arg0, %arg2, %arg2)
    (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>,
     %arg6: tensor<4x8xf32>, %arg7: tensor<4x8xf32>, %arg8: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
    %2 = stablehlo.multiply %arg5, %arg6 : tensor<4x8xf32>
    %3 = stablehlo.subtract %1, %arg7 : tensor<4x8xf32>
    %4 = stablehlo.divide %2, %arg8 : tensor<4x8xf32>
    %5 = stablehlo.power %3, %4 : tensor<4x8xf32>
    mpmd.return %5 : tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
       tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
      -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @duplicate_and_noop_results
func.func @duplicate_and_noop_results(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: %[[NC:.*]]:2 = mpmd.named_computation<"f"> (%arg0, %arg1)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:   %[[MUL:.*]] = stablehlo.multiply %arg2, %arg3
// CHECK-NEXT:   mpmd.return %[[ADD]], %[[MUL]]
// CHECK-NEXT: }
// CHECK-NEXT: return %arg0, %[[NC]]#0, %[[NC]]#1,
// CHECK-SAME:        %[[NC]]#0, %arg0, %[[NC]]#0
  %0:6 = mpmd.named_computation<"f"> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    %2 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %arg2, %1, %2, %1, %arg2, %1 :
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>,
          tensor<4x8xf32>, tensor<4x8xf32>,
          tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 :
    tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
    tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @duplicate_operands_and_results
func.func @duplicate_operands_and_results(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: %[[NC:.*]]:2 = mpmd.named_computation<"f"> (%arg0, %arg1)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[ADD]], %arg2
// CHECK-NEXT:   mpmd.return %[[ADD]], %[[MUL]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[NC]]#0, %[[NC]]#1, %[[NC]]#1
  %0:3 = mpmd.named_computation<"f"> (%arg0, %arg1, %arg0)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    %2 = stablehlo.multiply %1, %arg4 : tensor<4x8xf32>
    mpmd.return %1, %2, %2 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %0#0, %0#1, %0#2 :
    tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @unused_operand
func.func @unused_operand(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> tensor<4x8xf32>
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: %[[NC:.*]] = mpmd.named_computation<"f"> (%arg0)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[NC]]
  %0 = mpmd.named_computation<"f"> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @unused_result
func.func @unused_result(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: %[[NC:.*]] = mpmd.named_computation<"f"> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[NC]]
  %0:2 = mpmd.named_computation<"f"> (%arg0)
    (%arg2: tensor<4x8xf32>) {
      %1 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      // This value is not used outside the named computation. It'll be removed.
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %0#0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @unused_result_causes_operand_to_be_removed
func.func @unused_result_causes_operand_to_be_removed(%arg0: tensor<4x8xf32>)
  -> tensor<4x8xf32>
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: %[[NC:.*]] = mpmd.named_computation<"f"> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[NC]]
  %0 = mpmd.named_computation<"f"> (%arg0)
    (%arg2: tensor<4x8xf32>) {
      mpmd.return %arg2 : tensor<4x8xf32>
    } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %1:2 = mpmd.named_computation<"f"> (%arg0, %0)
    (%arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) {
      %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
      // This value is not used outside the named computation. It'll be removed
      // and so it the named computation that produces the operand re to %arg2.
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %1#0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @noop_named_computation
func.func @noop_named_computation(%arg0: tensor<4x8xf32>)
  -> tensor<4x8xf32>
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: return %arg0
  %0 = mpmd.named_computation<"f"> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @noop_result_with_opt_barrier
func.func @noop_result_with_opt_barrier(%arg0: tensor<4x8xf32>)
  -> tensor<4x8xf32>
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: %[[NC:.*]] = mpmd.named_computation<"f"> (%arg0)
// CHECK-NEXT:   %[[OPTB:.*]] = stablehlo.optimization_barrier %arg1
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[OPTB]], %[[OPTB]] : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[NC]]
  %0:2 = mpmd.named_computation<"f"> (%arg0, %arg0)
    (%arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) {
    %optb:2  = stablehlo.optimization_barrier %arg1, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>
    %1 = stablehlo.add %optb#0, %optb#0 : tensor<4x8xf32>
    mpmd.return %1, %optb#1 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %0#0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @noop_result_with_opt_barrier_multiple_uses_not_simplified
func.func @noop_result_with_opt_barrier_multiple_uses_not_simplified(
  %arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: %[[NC:.*]]:2 = mpmd.named_computation<"f"> (%arg0, %arg1)
// CHECK-NEXT:   %[[OPTB:.*]]:2 = stablehlo.optimization_barrier %arg2, %arg3
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[OPTB]]#0, %[[OPTB]]#0 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]], %[[OPTB]]#1
// CHECK-NEXT: }
// CHECK-NEXT: return %[[NC]]#0, %[[NC]]#1
  %0:3 = mpmd.named_computation<"f"> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %optb:2  = stablehlo.optimization_barrier %arg2, %arg3 : tensor<4x8xf32>, tensor<4x8xf32>
    %1 = stablehlo.add %optb#0, %optb#0 : tensor<4x8xf32>
    mpmd.return %1, %optb#1, %optb#1 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %0#0, %0#1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @noop_results_and_duplicate_operand
func.func @noop_results_and_duplicate_operand(
  %arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
// CHECK-NEXT: %[[NC:.*]] = mpmd.named_computation<"f"> (%arg1)
// CHECK-SAME:   (%arg3: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg3, %arg3 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %arg0, %arg0, %[[NC]], %arg2
  %0:4 = mpmd.named_computation<"f"> (%arg0, %arg0, %arg1, %arg2)
    (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>,
     %arg6: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg5, %arg5 : tensor<4x8xf32>
    mpmd.return %arg3, %arg4, %1, %arg6 :
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %0#0, %0#1, %0#2, %0#3 :
    tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
}

// The all_gather cannot be removed because it isn't pure.
// CHECK-LABEL: func @all_results_unused_but_not_pure
func.func @all_results_unused_but_not_pure(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
  // CHECK-NEXT: %[[NC:.*]] = mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>) {
  // CHECK-NEXT:   %[[AG:.*]] = "stablehlo.all_gather"(%arg1)
  // CHECK-NEXT:   mpmd.return %1 : tensor<4x16xf32>
  // CHECK-NEXT: } : (tensor<4x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: return %arg0
  %0:3 = mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>) {
    %1 = "stablehlo.all_gather"(%arg1) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    } : (tensor<4x8xf32>) -> tensor<4x16xf32>
    %2 = stablehlo.add %1, %1 : tensor<4x16xf32>
    mpmd.return %arg1, %1, %2 : tensor<4x8xf32>, tensor<4x16xf32>, tensor<4x16xf32>
  } : (tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x16xf32>, tensor<4x16xf32>)
  return %arg0 : tensor<4x8xf32>
}
