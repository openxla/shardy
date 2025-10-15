// RUN: mpmd_opt %s -mpmd-simplify-program -mlir-diagnostic-verbosity-level=errors 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_dist_x = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
!mesh_1_tensor_dist_y = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"y"}, {?}]>>

// CHECK-LABEL: func @no_duplicate_operands_or_results
func.func @no_duplicate_operands_or_results(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %2 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

// CHECK-LABEL: func @duplicate_operands
func.func @duplicate_operands(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor, %arg2: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1, %arg2)
// CHECK-SAME:   (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg3, %arg4
// CHECK-NEXT:   %[[MUL:.*]] = stablehlo.multiply %arg5, %arg3
// CHECK-NEXT:   %[[SUB:.*]] = stablehlo.subtract %[[ADD]], %arg5
// CHECK-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[MUL]], %arg5
// CHECK-NEXT:   %[[POW:.*]] = stablehlo.power %[[SUB]], %[[DIV]]
// CHECK-NEXT:   mpmd.return %[[POW]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1, %arg2, %arg0, %arg2, %arg2)
    (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>,
     %arg6: tensor<4x8xf32>, %arg7: tensor<4x8xf32>, %arg8: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
    %2 = stablehlo.multiply %arg5, %arg6 : tensor<4x8xf32>
    %3 = stablehlo.subtract %1, %arg7 : tensor<4x8xf32>
    %4 = stablehlo.divide %2, %arg8 : tensor<4x8xf32>
    %5 = stablehlo.power %3, %4 : tensor<4x8xf32>
    mpmd.return %5 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor,
       !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
      -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

// CHECK-LABEL: func @duplicate_and_noop_results
func.func @duplicate_and_noop_results(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor,
      !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
    attributes { "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:   %[[MUL:.*]] = stablehlo.multiply %arg2, %arg3
// CHECK-NEXT:   mpmd.return %[[ADD]], %[[MUL]]
// CHECK-NEXT: }
// CHECK-NEXT: return %arg0, %[[FRAGMENT]]#0, %[[FRAGMENT]]#1,
// CHECK-SAME:        %[[FRAGMENT]]#0, %arg0, %[[FRAGMENT]]#0
  %0:6 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    %2 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %arg2, %1, %2, %1, %arg2, %1 :
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor)
      -> (!mesh_1_tensor, !mesh_1_tensor,
          !mesh_1_tensor, !mesh_1_tensor,
          !mesh_1_tensor, !mesh_1_tensor)
  func.return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 :
    !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor,
    !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// CHECK-LABEL: func @duplicate_results_with_mismatched_types
func.func @duplicate_results_with_mismatched_types(%arg0: !mesh_1_tensor_dist_x, %arg1: !mesh_1_tensor_dist_x)
  -> (!mesh_1_tensor_dist_x, !mesh_1_tensor_dist_x, !mesh_1_tensor_dist_y,
      !mesh_1_tensor_dist_x, !mesh_1_tensor_dist_x, !mesh_1_tensor_dist_y)
    attributes { "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]]:3 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:   %[[MUL:.*]] = stablehlo.multiply %arg2, %arg3
// CHECK-NEXT:   mpmd.return %[[ADD]], %[[MUL]], %[[ADD]]
// CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>,
// CHECK-SAME:      !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
// CHECK-SAME:     -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>,
// CHECK-SAME:         !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>,
// CHECK-SAME:         !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"y"}, {?}]>>)
// CHECK-NEXT: return %[[FRAGMENT]]#0, %[[FRAGMENT]]#1, %[[FRAGMENT]]#2
// CHECK-SAME:        %[[FRAGMENT]]#0, %[[FRAGMENT]]#0, %[[FRAGMENT]]#2
  %0:6 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    %2 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %1, %2, %1, %1, %1, %1 :
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_dist_x, !mesh_1_tensor_dist_x)
      -> (!mesh_1_tensor_dist_x, !mesh_1_tensor_dist_x,
          !mesh_1_tensor_dist_y, !mesh_1_tensor_dist_x,
          !mesh_1_tensor_dist_x, !mesh_1_tensor_dist_y)
  func.return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 :
    !mesh_1_tensor_dist_x, !mesh_1_tensor_dist_x, !mesh_1_tensor_dist_y,
    !mesh_1_tensor_dist_x, !mesh_1_tensor_dist_x, !mesh_1_tensor_dist_y
}

// CHECK-LABEL: func @duplicate_operands_and_results
func.func @duplicate_operands_and_results(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[ADD]], %arg2
// CHECK-NEXT:   mpmd.return %[[ADD]], %[[MUL]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]#0, %[[FRAGMENT]]#1, %[[FRAGMENT]]#1
  %0:3 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1, %arg0)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    %2 = stablehlo.multiply %1, %arg4 : tensor<4x8xf32>
    mpmd.return %1, %2, %2 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
      -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
  func.return %0#0, %0#1, %0#2 :
    !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// CHECK-LABEL: func @unused_operand
func.func @unused_operand(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

// CHECK-LABEL: func @unused_result
func.func @unused_result(%arg0: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]
  %0:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
      %1 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      // This value is not used outside the fragment. It will be removed.
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (!mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)
  func.return %0#0 : !mesh_1_tensor
}

// CHECK-LABEL: func @unused_result_causes_operand_to_be_removed
func.func @unused_result_causes_operand_to_be_removed(%arg0: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
      mpmd.return %arg2 : tensor<4x8xf32>
    } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %0)
    (%arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) {
      %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
      // This value is not used outside the fragment. It will be removed and
      // so it the fragment that produces the operand re to %arg2.
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (!mesh_1_tensor, !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)
  func.return %1#0 : !mesh_1_tensor
}

// CHECK-LABEL: func @noop_fragment
func.func @noop_fragment(%arg0: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: return %arg0
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

// CHECK-LABEL: func @noop_fragment_type_mismatch
func.func @noop_fragment_type_mismatch(%arg0: !mesh_1_tensor_dist_x)
  -> !mesh_1_tensor_dist_y attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   mpmd.return %arg1
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_dist_x) -> !mesh_1_tensor_dist_y
  func.return %0 : !mesh_1_tensor_dist_y
}

// CHECK-LABEL: func @noop_results_and_duplicate_operand
func.func @noop_results_and_duplicate_operand(
  %arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor, %arg2: !mesh_1_tensor, %arg3: !mesh_1_tensor_dist_x)
  -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor_dist_y) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg1, %arg3)
// CHECK-SAME:   (%arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg4, %arg4 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]], %arg5
// CHECK-NEXT: }
// CHECK-NEXT: return %arg0, %arg0, %[[FRAGMENT]]#0, %arg2, %[[FRAGMENT]]#1
  %0:5 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg0, %arg1, %arg2, %arg3)
    (%arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>, %arg6: tensor<4x8xf32>,
     %arg7: tensor<4x8xf32>, %arg8: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg6, %arg6 : tensor<4x8xf32>
    mpmd.return %arg4, %arg5, %1, %arg7, %arg8 :
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
      tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor,
       !mesh_1_tensor, !mesh_1_tensor_dist_x)
      -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor,
          !mesh_1_tensor, !mesh_1_tensor_dist_y)
  func.return %0#0, %0#1, %0#2, %0#3, %0#4 :
    !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor,
    !mesh_1_tensor, !mesh_1_tensor_dist_y
}
