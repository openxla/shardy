// RUN: mpmd_opt %s -mpmd-rule-based-merge='rules=FragmentMergeRule(sources=[FragmentInfo(origins=["f"],mesh_name="m1"),FragmentInfo(origins=["g"],mesh_name="m1")],target=FragmentInfo(origins=["f","g"],mesh_name="m1")),FragmentMergeRule(sources=[FragmentInfo(origins=["i"],mesh_name="m1"),FragmentInfo(origins=["j"],mesh_name="m1"),FragmentInfo(origins=["k"],mesh_name="m1")],target=FragmentInfo(origins=["i","j","k"],mesh_name="m1"))' 2>&1 | FileCheck %s
// TODO(b/435182733) Add more lit tests for rule based merge pass.

!mesh_1_tensor_2_2_f32 = !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>

// CHECK-LABEL: func @merge_two_fragments
func.func @merge_two_fragments
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
  %2 = mpmd.fragment<mesh="m1", origin=["g"]> (%0, %arg2) (%arg3: tensor<2x2xf32>, %arg4: tensor<2x2xf32>) {
    %3 = stablehlo.add %arg3, %arg4 : tensor<2x2xf32>
    mpmd.return %3 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  // CHECK-NEXT: return %[[RES]] : !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  return %2 : !mesh_1_tensor_2_2_f32
}

// CHECK-LABEL: func @merge_three_fragments
func.func @merge_three_fragments
(%arg0: !mesh_1_tensor_2_2_f32, %arg1: !mesh_1_tensor_2_2_f32, %arg2: !mesh_1_tensor_2_2_f32)
 -> (!mesh_1_tensor_2_2_f32)
 attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK: %[[FRAG1_RES:.*]] = mpmd.fragment<mesh="m1", origin=["i", "j", "k"]> (%arg0, %arg1, %arg2) (%arg3: tensor<2x2xf32>, %arg4: tensor<2x2xf32>, %arg5: tensor<2x2xf32>) {
  // CHECK-NEXT:   %2 = stablehlo.add %arg3, %arg4 : tensor<2x2xf32>
  // CHECK-NEXT:   %3 = stablehlo.add %2, %arg5 : tensor<2x2xf32>
  // CHECK-NEXT:   %4 = stablehlo.add %3, %arg4 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %4 : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  %0 = mpmd.fragment<mesh="m1", origin=["i"]> (%arg0, %arg1) (%arg3: tensor<2x2xf32>, %arg4: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg4 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  %2 = mpmd.fragment<mesh="m1", origin=["j"]> (%0, %arg2) (%arg3: tensor<2x2xf32>, %arg4: tensor<2x2xf32>) {
    %3 = stablehlo.add %arg3, %arg4 : tensor<2x2xf32>
    mpmd.return %3 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  %4 = mpmd.fragment<mesh="m1", origin=["k"]> (%2, %arg1) (%arg3: tensor<2x2xf32>, %arg4: tensor<2x2xf32>) {
    %5 = stablehlo.add %arg3, %arg4 : tensor<2x2xf32>
    mpmd.return %5 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  // CHECK: %[[FRAG2_RES:.*]] = mpmd.fragment<mesh="m1", origin=["no_merge"]> (%0, %arg2) (%arg3: tensor<2x2xf32>, %arg4: tensor<2x2xf32>) {
  // CHECK-NEXT:   %2 = stablehlo.add %arg3, %arg4 : tensor<2x2xf32>
  // CHECK-NEXT:   mpmd.return %2 : tensor<2x2xf32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<2x2xf32>>, !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  %6 = mpmd.fragment<mesh="m1", origin=["no_merge"]> (%4, %arg2) (%arg3: tensor<2x2xf32>, %arg4: tensor<2x2xf32>) {
    %7 = stablehlo.add %arg3, %arg4 : tensor<2x2xf32>
    mpmd.return %7 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  // CHECK-NEXT: return %[[FRAG2_RES]] : !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>
  return %6 : !mesh_1_tensor_2_2_f32
}
