// RUN: mpmd_opt %s -mpmd-rule-based-merge=rules='FragmentMergeRule(sources=[FragmentInfo(origins=["f"]),FragmentInfo(origins=["g"])],target=FragmentInfo(origins=["f","g"]))' 2>&1 | FileCheck %s
// TODO(b/435182733) Add more lit tests for rule based merge pass.

!mesh_1_tensor_2_2_f32 = !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>

// CHECK-LABEL: func @main
func.func @main
(%arg0: !mesh_1_tensor_2_2_f32, %arg1: !mesh_1_tensor_2_2_f32, %arg2: !mesh_1_tensor_2_2_f32)
 -> (!mesh_1_tensor_2_2_f32)
 attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK: %[[RES:.*]] = mpmd.fragment<mesh="m1", origin=["f", "g"]> (%arg0, %arg1, %arg2) (%arg3: tensor<2x2xf32>, %arg4: tensor<2x2xf32>, %arg5: tensor<2x2xf32>) {
  // CHECK-NEXT:   %1 = stablehlo.add %arg3, %arg4 : tensor<2x2xf32>
  // CHECK-NEXT:   %2 = stablehlo.add %1, %arg5 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %2 : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1) (%arg3: tensor<2x2xf32>, %arg4: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg4 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  %2 = mpmd.fragment<mesh="m1", origin=["g"]> (%0, %arg2) (%arg5: tensor<2x2xf32>, %arg6: tensor<2x2xf32>) {
    %3 = stablehlo.add %arg5, %arg6 : tensor<2x2xf32>
    mpmd.return %3 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  // CHECK-NEXT: return %[[RES]] : !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  return %2 : !mesh_1_tensor_2_2_f32
}
