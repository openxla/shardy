// RUN: mpmd_opt %s -mpmd-split-bwd-fragments -mpmd-fragment-dedup -mpmd-fragment-dce -canonicalize -mpmd-fragment-dce 2>&1 | FileCheck %s

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

// CHECK-LABEL: @split_simple
func.func public @split_simple(%arg0: !mesh1_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh1_16x64_f32) -> (!mesh0_16x16_f32, !mesh1_16x64_f32)
  attributes {topology = #mpmd.topology<<"mesh0" : <["x"=1]>>, <"mesh1" : <["x"=1]>>>} {
    // CHECK:     %[[F1:.*]] = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1) {split_keep_transferred}
    // CHECK-NEXT:   %[[D1:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_one}
    // CHECK-NEXT:   mpmd.return %[[D1]]

    // CHECK:     %[[F2:.*]] = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg2, %[[F1]]) {split_drop_transferred}
    // CHECK-NEXT:   %[[D2:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_two}
    // CHECK-NEXT:   mpmd.return %[[D2]]

    // CHECK:     %[[T:.*]] = mpmd.transfer %[[F1]]
    // CHECK-NEXT: return %[[T]], %[[F2]]
    %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
      (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
        %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
        %2 = "stablehlo.dot"(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
        mpmd.return %1, %2 : tensor<16x16xf32>, tensor<16x64xf32>
      } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
    %4 = mpmd.transfer %0#0 : (!mesh1_16x16_f32) -> !mesh0_16x16_f32
    func.return %4, %0#1 : !mesh0_16x16_f32, !mesh1_16x64_f32
}

// This is exactly like `split_simple` but contains the second dot inside a
// region. As a result %1 does not appear directly as operand of %2, rather it
// is treated as an extra operand (see `GetEffectiveOperands()` in
// split_bwd_fragments.cc)
// CHECK-LABEL: @split_simple_region
func.func public @split_simple_region(%arg0: !mesh1_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh1_16x64_f32) -> (!mesh0_16x16_f32, !mesh1_16x64_f32)
  attributes {topology = #mpmd.topology<<"mesh0" : <["x"=1]>>, <"mesh1" : <["x"=1]>>>} {
    // CHECK:     %[[F1:.*]] = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1) {split_keep_transferred}
    // CHECK:       %[[D1:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_one}
    // CHECK-NEXT:  mpmd.return %[[D1]]

    // CHECK:     %[[F2:.*]] = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%[[F1]], %arg2) {split_drop_transferred}
    // CHECK:     %[[WHILE:.*]] = stablehlo.while
    // CHECK:        %[[D2:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_two}
    // CHECK-NEXT:     stablehlo.return %[[D2]]
    // CHECK:        mpmd.return %[[WHILE]]

    // CHECK:     %[[T:.*]] = mpmd.transfer %[[F1]]
    // CHECK-NEXT: return %[[T]], %[[F2]]
    %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
      (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
        %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>

        %2 = stablehlo.while(%iterArg = %arg12) : tensor<16x64xf32>
             cond {
               %3 = stablehlo.constant dense<true> : tensor<i1>
               stablehlo.return %3 : tensor<i1>
             } do {
               %4 = "stablehlo.dot"(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
               stablehlo.return %4 : tensor<16x64xf32>
             }
        mpmd.return %1, %2 : tensor<16x16xf32>, tensor<16x64xf32>
      } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
    %4 = mpmd.transfer %0#0 : (!mesh1_16x16_f32) -> !mesh0_16x16_f32
    func.return %4, %0#1 : !mesh0_16x16_f32, !mesh1_16x64_f32
}


// CHECK-LABEL: @no_transfer_no_split
func.func public @no_transfer_no_split(%arg0: !mesh1_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
  attributes {topology = #mpmd.topology<<"mesh0" : <["x"=1]>>, <"mesh1" : <["x"=1]>>>} {
   // CHECK: %[[F:.*]]:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
   // CHECK-NEXT: %[[D1:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_one}
   // CHECK-NEXT: %[[D2:.*]] = stablehlo.dot %[[D1]], %[[_:.*]] {dot_two}
   // CHECK-NEXT: mpmd.return %[[D1]], %[[D2]]
   // CHECK: return %[[F]]#0, %[[F]]#1
    %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
      (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
        %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
        %2 = "stablehlo.dot"(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
        mpmd.return %1, %2 : tensor<16x16xf32>, tensor<16x64xf32>
      } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
    func.return %0#0, %0#1 : !mesh1_16x16_f32, !mesh1_16x64_f32
}

// CHECK-LABEL: @only_transfer_no_split
func.func public @only_transfer_no_split(%arg0: !mesh1_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh1_16x64_f32) -> (!mesh0_16x16_f32, !mesh0_16x64_f32)
  attributes {topology = #mpmd.topology<<"mesh0" : <["x"=1]>>, <"mesh1" : <["x"=1]>>>} {
   // CHECK: %[[F:.*]]:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
   // CHECK-NEXT: %[[D1:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_one}
   // CHECK-NEXT: %[[D2:.*]] = stablehlo.dot %[[D1]], %[[_:.*]] {dot_two}
   // CHECK-NEXT: mpmd.return %[[D1]], %[[D2]]
   // CHECK-DAG: %[[T1:.*]] = mpmd.transfer %[[F]]#0
   // CHECK-DAG: %[[T2:.*]] = mpmd.transfer %[[F]]#1
   // CHECK: return %[[T1]], %[[T2]]
    %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
      (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
        %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
        %2 = "stablehlo.dot"(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
        mpmd.return %1, %2 : tensor<16x16xf32>, tensor<16x64xf32>
      } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
    %4 = mpmd.transfer %0#0 : (!mesh1_16x16_f32) -> !mesh0_16x16_f32
    %5 = mpmd.transfer %0#1 : (!mesh1_16x64_f32) -> !mesh0_16x64_f32
    func.return %4, %5 : !mesh0_16x16_f32, !mesh0_16x64_f32
}


// The following test shows how the split fragments contain split ops, whether
// they have side-effects or not. This is, arguably, somewhat questionable as
// we do not directly have knowledge of exactly what side-effects the fragment
// ops perform, but it is a behaviour we adopt in other parts too (e.g. merging)
// CHECK-LABEL: @split_simple_with_side_effects
func.func public @split_simple_with_side_effects(%arg0: !mesh1_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh1_16x64_f32) -> (!mesh0_16x16_f32, !mesh1_16x64_f32)
  attributes {topology = #mpmd.topology<<"mesh0" : <["x"=1]>>, <"mesh1" : <["x"=1]>>>} {
    // CHECK: mpmd.fragment
    // CHECK-NEXT: stablehlo.dot
    // CHECK-NEXT: mpmd.return
    // CHECK: mpmd.fragment
    // CHECK-NEXT: stablehlo.custom_call @custom_dot
    %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
      (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
        %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
        %2 = stablehlo.custom_call @custom_dot(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
        mpmd.return %1, %2 : tensor<16x16xf32>, tensor<16x64xf32>
      } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
    %4 = mpmd.transfer %0#0 : (!mesh1_16x16_f32) -> !mesh0_16x16_f32
    func.return %4, %0#1 : !mesh0_16x16_f32, !mesh1_16x64_f32
}

// CHECK-LABEL: @split_simple_cant_pull
func.func public @split_simple_cant_pull(%arg0: !mesh1_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh0_16x64_f32)
  attributes {topology = #mpmd.topology<<"mesh0" : <["x"=1]>>, <"mesh1" : <["x"=1]>>>} {
   // CHECK: %[[F:.*]]:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
   // CHECK-NEXT: %[[D1:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_one}
   // CHECK-NEXT: %[[D2:.*]] = stablehlo.dot %[[D1]], %[[_:.*]] {dot_two}
   // CHECK-NEXT: mpmd.return %[[D1]], %[[D2]]
    %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
      (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
        %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
        %2 = "stablehlo.dot"(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
        mpmd.return %1, %2 : tensor<16x16xf32>, tensor<16x64xf32>
      } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32)
    // We transfer %0#1 so we will attempt to split %0#0 out, but it is not associated with any
    // computation that can be pulled, so the split fragment will just disappear after simplification.
    %4 = mpmd.transfer %0#1 : (!mesh1_16x64_f32) -> !mesh0_16x64_f32
    func.return %0#0, %4 : !mesh1_16x16_f32, !mesh0_16x64_f32
}

// CHECK-LABEL: @split_simple_multiple_matmuls
func.func public @split_simple_multiple_matmuls(%arg0: !mesh1_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh1_16x64_f32) -> (!mesh0_16x16_f32, !mesh1_8x64_f32)
  attributes {topology = #mpmd.topology<<"mesh0" : <["x"=1]>>, <"mesh1" : <["x"=1]>>>} {
    // CHECK:     %[[F1:.*]] = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1) {split_keep_transferred}
    // CHECK-NEXT:   %[[D1:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_one}
    // CHECK-NEXT:   mpmd.return %[[D1]]

    // CHECK:     %[[F2:.*]] = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg2, %[[F1]], %arg1) {split_drop_transferred}
    // CHECK-DAG:   %[[D2:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_two}
    // CHECK-DAG:   %[[D3:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_three}
    // CHECK-DAG:   mpmd.return %[[D3]]
    %0:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
      (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
        %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
        %2 = "stablehlo.dot"(%1, %arg12) {dot_two}: (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
        %3 = "stablehlo.dot"(%arg11, %2) {dot_three}: (tensor<8x16xf32>, tensor<16x64xf32>) -> tensor<8x64xf32>
        mpmd.return %1, %3 : tensor<16x16xf32>, tensor<8x64xf32>
      } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_8x64_f32)
    %4 = mpmd.transfer %0#0 : (!mesh1_16x16_f32) -> !mesh0_16x16_f32
    func.return %4, %0#1 : !mesh0_16x16_f32, !mesh1_8x64_f32
}

// CHECK-LABEL: @split_simple_multiple_transfers
func.func public @split_simple_multiple_transfers(%arg0: !mesh1_16x8_f32, %arg1 : !mesh1_8x16_f32, %arg2: !mesh1_16x64_f32) -> (!mesh0_16x16_f32, !mesh0_16x64_f32, !mesh1_8x64_f32)
  attributes {topology = #mpmd.topology<<"mesh0" : <["x"=1]>>, <"mesh1" : <["x"=1]>>>} {
    // CHECK:     %[[F1:.*]]:2 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2) {split_keep_transferred}
    // CHECK-NEXT:   %[[D1:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_one}
    // CHECK-NEXT:   %[[D2:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_two}
    // CHECK-NEXT:   mpmd.return %[[D1]], %[[D2]]

    // CHECK:     %[[F2:.*]] = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%[[F1]]#1, %arg1) {split_drop_transferred}
    // CHECK-NEXT:   %[[D3:.*]] = stablehlo.dot %[[_:.*]], %[[_:.*]] {dot_three}
    // CHECK-NEXT:   mpmd.return %[[D3]]
    %0:3 = mpmd.fragment<mesh="mesh1", origin=["block"(1)]> (%arg0, %arg1, %arg2)
      (%arg10: tensor<16x8xf32>, %arg11 : tensor<8x16xf32>, %arg12: tensor<16x64xf32>) {
        %1 = "stablehlo.dot"(%arg10, %arg11) {dot_one} : (tensor<16x8xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
        %2 = "stablehlo.dot"(%1, %arg12) {dot_two} : (tensor<16x16xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
        %3 = "stablehlo.dot"(%arg11, %2) {dot_three} : (tensor<8x16xf32>, tensor<16x64xf32>) -> tensor<8x64xf32>
        mpmd.return %1, %2, %3 : tensor<16x16xf32>, tensor<16x64xf32>, tensor<8x64xf32>
      } : (!mesh1_16x8_f32, !mesh1_8x16_f32, !mesh1_16x64_f32) -> (!mesh1_16x16_f32, !mesh1_16x64_f32, !mesh1_8x64_f32)
    // Here by transferring two values, we block the pulling of {dot_two}
    %4 = mpmd.transfer %0#0 : (!mesh1_16x16_f32) -> !mesh0_16x16_f32
    %5 = mpmd.transfer %0#1 : (!mesh1_16x64_f32) -> !mesh0_16x64_f32
    func.return %4, %5, %0#2 : !mesh0_16x16_f32, !mesh0_16x64_f32, !mesh1_8x64_f32
}


// This is one of the backward fragments of a two-layer MLP.
// CHECK-LABEL: @split_bwd_mlp
func.func public @split_bwd_mlp(%arg0: !mesh1_8x8_i1, %arg1: !mesh1_8x8_f32, %arg2: !mesh1_8x8_f32, %arg3: !mesh1_8x8_i1, %arg4: !mesh1_8x8_f32, %arg5: !mesh1_8x8_f32) -> (!mesh0_8x8_f32, !mesh1_8x8_f32, !mesh1_8x8_f32)
  attributes {topology = #mpmd.topology<<"mesh0" : <["x"=1]>>, <"mesh1" : <["x"=1]>>>} {
    // CHECK-NEXT: %[[FTRAN:.*]]:3 = mpmd.fragment<mesh="mesh1", origin=["layer_3"(1), "layer_2"(1)]> (%arg0, %arg2, %arg3, %arg5) {call_counter = 0 : ui32, split_keep_transferred}
    // CHECK-DAG: %[[S1:.*]] = stablehlo.select %[[_:.*]], %[[_:.*]], %[[_:.*]]
    // CHECK-DAG: %[[D1:.*]] = stablehlo.dot_general %[[S1]], %[[_:.*]]
    // CHECK-DAG: %[[T1:.*]] = stablehlo.transpose %[[D1]]
    // CHECK-DAG: %[[S2:.*]] = stablehlo.select %[[_:.*]], %[[T1]], %[[_:.*]]
    // CHECK-DAG: %[[D2:.*]] = stablehlo.dot_general %[[S2]], %[[_:.*]]
    // CHECK-DAG: %[[T2:.*]] = stablehlo.transpose %[[D2]]
    // CHECK-DAG: mpmd.return %[[T2]], %[[S2]], %[[S1]]

    // CHECK: %[[FRES:.*]]:2 = mpmd.fragment<mesh="mesh1", origin=["layer_3"(1), "layer_2"(1)]> (%arg4, %[[FTRAN]]#1, %arg1, %[[FTRAN]]#2) {call_counter = 0 : ui32, split_drop_transferred}
    // CHECK-DAG: %[[G1:.*]] = stablehlo.dot_general %arg9, %arg8
    // CHECK-DAG: %[[G2:.*]] = stablehlo.dot_general %arg7, %arg6
    // CHECK-DAG: mpmd.return %[[G1]], %[[G2]] : tensor<8x8xf32>, tensor<8x8xf32>

    // CHECK: %[[TRANSFER:.*]] = mpmd.transfer %[[FTRAN]]#0
    // CHECK-DAG: %[[TRANSFER]], %[[FRES]]#0, %[[FRES]]#1
    %4:3 = mpmd.fragment<mesh="mesh1", origin=["layer_3"(1), "layer_2"(1)]> (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {call_counter = 0 : ui32}
    (%arg11: tensor<8x8xi1>, %arg6: tensor<8x8xf32>, %arg7: tensor<8x8xf32>, %arg8: tensor<8x8xi1>, %arg9: tensor<8x8xf32>, %arg10: tensor<8x8xf32>) {
      %9 = stablehlo.constant dense<1.000000e+00> : tensor<8x8xf32>
      %10 = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf32>
      %11 = stablehlo.select %arg11, %9, %10 : tensor<8x8xi1>, tensor<8x8xf32>
      %12 = "stablehlo.dot_general"(%11, %arg7) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}> : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
      %13 = stablehlo.transpose %12, dims = [1,0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
      %14 = "stablehlo.dot_general"(%11, %arg6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}> : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
      %15 = stablehlo.constant dense<0.000000e+00> : tensor<8x8xf32>
      %16 = stablehlo.select %arg8, %13, %15 : tensor<8x8xi1>, tensor<8x8xf32>
      %17 = "stablehlo.dot_general"(%16, %arg10) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}> : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
      %18 = stablehlo.transpose %17, dims = [1,0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
      %19 = "stablehlo.dot_general"(%16, %arg9) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}> : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
      mpmd.return %14, %19, %18 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
    } : (!mesh1_8x8_i1, !mesh1_8x8_f32, !mesh1_8x8_f32, !mesh1_8x8_i1, !mesh1_8x8_f32, !mesh1_8x8_f32) -> (!mesh1_8x8_f32, !mesh1_8x8_f32, !mesh1_8x8_f32)
    %5 = mpmd.transfer %4#2 : (!mesh1_8x8_f32) -> !mesh0_8x8_f32
    func.return %5, %4#0, %4#1 : !mesh0_8x8_f32, !mesh1_8x8_f32, !mesh1_8x8_f32
}
