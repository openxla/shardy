// RUN: mpmd_opt %s -mpmd-fragment-dedup 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_dist_x = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
!mesh_1_tensor_dist_y = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"y"}, {?}]>>
#topology =#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>

module {

// CHECK-LABEL: func @duplicate_operands
func.func @duplicate_operands(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor, %arg2: !mesh_1_tensor)
  -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
  attributes {"topology"=#topology} {
  %0:3 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1, %arg2, %arg0, %arg2, %arg2)
    (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>,
     %arg6: tensor<4x8xf32>, %arg7: tensor<4x8xf32>, %arg8: tensor<4x8xf32>) {
    // %arg3, %arg4, %arg5 are the unique operands. They should be used instead of
    // %arg6, %arg7, %arg8.
    // CHECK: %1 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
    // CHECK: %2 = stablehlo.add %arg5, %arg3 : tensor<4x8xf32>
    // CHECK: %3 = stablehlo.add %arg5, %arg5 : tensor<4x8xf32>
    %1 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
    %2 = stablehlo.add %arg5, %arg6 : tensor<4x8xf32>
    %3 = stablehlo.add %arg7, %arg8 : tensor<4x8xf32>
    mpmd.return %1, %2, %3 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor,
       !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
      -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
  func.return %0#0, %0#1, %0#2 : !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// CHECK-LABEL: func @duplicate_results
func.func @duplicate_results(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
    attributes { "topology"=#topology} {
  // CHECK: %[[FRAGMENT:.*]]:4 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
  %0:4 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %1, %1, %1, %1 :
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
      tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor)
      -> (!mesh_1_tensor, !mesh_1_tensor,
          !mesh_1_tensor, !mesh_1_tensor)
  // Only use the first result since all the others are duplicates.
  // CHECK: %[[FRAGMENT]]#0, %[[FRAGMENT]]#0, %[[FRAGMENT]]#0, %[[FRAGMENT]]#0
  func.return %0#0, %0#1, %0#2, %0#3 :
    !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor,
    !mesh_1_tensor
}

// CHECK-LABEL: func @duplicate_results_but_with_different_types
func.func @duplicate_results_but_with_different_types(%arg0: !mesh_1_tensor_dist_x) -> (!mesh_1_tensor_dist_x, !mesh_1_tensor_dist_y)
  attributes {"topology"=#topology}
{
  // CHECK-NEXT: %[[F:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) (%arg1: tensor<4x8xf32>)
  // CHECK-NEXT:   mpmd.return %arg1, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[F]]#0, %[[F]]#1
  %0:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_dist_x) -> (!mesh_1_tensor_dist_x, !mesh_1_tensor_dist_y)
  func.return %0#0, %0#1 : !mesh_1_tensor_dist_x, !mesh_1_tensor_dist_y
}

}
