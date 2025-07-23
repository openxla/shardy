// RUN: mpmd_opt %s -mpmd-split-and-prioritize-transfer-independent-computations -canonicalize -mpmd-fragment-dedup -mpmd-fragment-dce 2>&1 \
// RUN:   | FileCheck  --implicit-check-not split_keep_transferred --implicit-check-not split_drop_transferred %s

!mesh1_8x8_i1 = !mpmd.mesh_tensor<"mesh1", tensor<8x8xi1>>
!mesh1_8x8_f32 = !mpmd.mesh_tensor<"mesh1", tensor<8x8xf32>>
!mesh0_8x8_f32 = !mpmd.mesh_tensor<"mesh0", tensor<8x8xf32>>
!mesh0_16x8_f32 = !mpmd.mesh_tensor<"mesh0", tensor<16x8xf32>>
!mesh0_8x16_f32 = !mpmd.mesh_tensor<"mesh0", tensor<8x16xf32>>
!mesh0_16x64_f32 = !mpmd.mesh_tensor<"mesh0", tensor<16x64xf32>>
!mesh0_8x64_f32 = !mpmd.mesh_tensor<"mesh0", tensor<8x64xf32>>
!mesh0_16x16_f32 = !mpmd.mesh_tensor<"mesh0", tensor<16x16xf32>>
!mesh1_16x8_f32 = !mpmd.mesh_tensor<"mesh1", tensor<16x8xf32>>
!mesh1_8x16_f32 = !mpmd.mesh_tensor<"mesh1", tensor<8x16xf32>>
!mesh1_16x64_f32 = !mpmd.mesh_tensor<"mesh1", tensor<16x64xf32>>
!mesh1_8x64_f32 = !mpmd.mesh_tensor<"mesh1", tensor<8x64xf32>>
!mesh1_16x16_f32 = !mpmd.mesh_tensor<"mesh1", tensor<16x16xf32>>
#topology = #mpmd.topology<<"mesh0" : <["x"=1]>>, <"mesh1" : <["x"=1]>>>



func.func public @split_simple(%arg0: !mesh1_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh0_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
  attributes {topology = #topology} {
  // CHECK:     %[[T:.*]] = mpmd.transfer %arg2
  // CHECK:     %[[F1:.*]] = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1) {split_keep_transferred}
  // CHECK-NEXT:   %[[D1:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_one}
  // CHECK-NEXT:   mpmd.return %[[D1]]
  // CHECK-NEXT: }

  // CHECK:     %[[F2:.*]] = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%[[T]], %[[F1]]) {split_drop_transferred}
  // CHECK-NEXT:   %[[D2:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_two}
  // CHECK-NEXT:   mpmd.return %[[D2]]
  // CHECK-NEXT: }

  // CHECK-NEXT: return %[[F1]], %[[F2]]
  %t2 = mpmd.transfer %arg2 : (!mesh0_16x64_f32) -> !mesh1_16x64_f32
  %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %t2)
    (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
      %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
      %2 = "stablehlo.dot"(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
      mpmd.return %1, %2 : tensor<16x16xf32>, tensor<16x64xf32>
    } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
  func.return %0#0, %0#1 : !mesh1_16x16_f32, !mesh1_16x64_f32
}



// CHECK-LABEL: @no_split_all_compute_relies_on_transferred_first_arg
func.func public @no_split_all_compute_relies_on_transferred_first_arg(%arg0: !mesh0_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
  attributes {topology = #topology} {
  // Only one fragment, since the split would pull out all the compute.
  // CHECK:     mpmd.fragment
  // CHECK-NOT: mpmd.fragment
  %t0 = mpmd.transfer %arg0 : (!mesh0_16x8_f32) -> !mesh1_16x8_f32
  %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%t0, %arg1, %arg2)
    (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
      %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
      %2 = "stablehlo.dot"(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
      mpmd.return %1, %2 : tensor<16x16xf32>, tensor<16x64xf32>
    } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
  func.return %0#0, %0#1 : !mesh1_16x16_f32, !mesh1_16x64_f32
}

// CHECK-LABEL: @no_split_only_one_arg_which_is_transferred
func.func public @no_split_only_one_arg_which_is_transferred(%arg0: !mesh0_16x8_f32) -> !mesh1_16x8_f32
  attributes {topology = #topology} {
  // Only one fragment, since the split would pull out all the compute.
  // CHECK:     mpmd.fragment
  // CHECK-NOT: mpmd.fragment
  %t0 = mpmd.transfer %arg0 : (!mesh0_16x8_f32) -> !mesh1_16x8_f32
  %0 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%t0)
    (%arg10: tensor<16x8xf32>) {
      %1 = stablehlo.add %arg10, %arg10 : tensor<16x8xf32>
      mpmd.return %1 : tensor<16x8xf32>
    } : (!mesh1_16x8_f32) -> !mesh1_16x8_f32
  func.return %0 : !mesh1_16x8_f32
}

// CHECK-LABEL: @no_transfer_no_split
func.func public @no_transfer_no_split(%arg0: !mesh1_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
  attributes {topology = #topology} {
    // CHECK:     mpmd.fragment
    // CHECK-NOT: mpmd.fragment
    %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
      (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
        %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
        %2 = "stablehlo.dot"(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
        mpmd.return %1, %2 : tensor<16x16xf32>, tensor<16x64xf32>
      } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
    func.return %0#0, %0#1 : !mesh1_16x16_f32, !mesh1_16x64_f32
}

// CHECK-LABEL: @only_transfer_no_split
func.func public @only_transfer_no_split(%arg0: !mesh0_16x8_f32, %arg1 : !mesh0_8x16_f32, %arg2: !mesh0_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
  attributes {topology = #topology} {
  // CHECK:     mpmd.fragment
  // CHECK-NOT: mpmd.fragment
  %t0 = mpmd.transfer %arg0 : (!mesh0_16x8_f32) -> !mesh1_16x8_f32
  %t1 = mpmd.transfer %arg1 : (!mesh0_8x16_f32) -> !mesh1_8x16_f32
  %t2 = mpmd.transfer %arg2 : (!mesh0_16x64_f32) -> !mesh1_16x64_f32
  %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%t0, %t1, %t2)
    (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
      %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
      %2 = "stablehlo.dot"(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
      mpmd.return %1, %2 : tensor<16x16xf32>, tensor<16x64xf32>
    } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
  func.return %0#0, %0#1 : !mesh1_16x16_f32, !mesh1_16x64_f32
}

