// RUN: mpmd_opt %s -mpmd-fragment-dedup -mpmd-fragment-dce 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
#topology =#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["y"=2]>>>
!mesh_1_tensor_dist_x = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
!mesh_1_tensor_dist_y = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"y"}, {?}]>>

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

// CHECK-LABEL: func @duplicate_results
func.func @duplicate_results(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor,
      !mesh_1_tensor)
    attributes { "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:   %[[MUL:.*]] = stablehlo.multiply %arg2, %arg3
// CHECK-NEXT:   mpmd.return %[[ADD]], %[[MUL]]
// CHECK-NEXT: }
  %0:4 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    %2 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %1, %2, %1, %2 :
      tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>,
      tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor)
      -> (!mesh_1_tensor, !mesh_1_tensor,
          !mesh_1_tensor, !mesh_1_tensor)
  func.return %0#0, %0#1, %0#2, %0#3 :
    !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor,
    !mesh_1_tensor
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
// CHECK-NEXT: return %[[FRAGMENT]]#0, %[[FRAGMENT]]#1
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
