// RUN: mpmd_opt %s -mpmd-mark-aliasing-and-donation 2>&1 | FileCheck --implicit-check-not arg_attrs %s


// These tests verify only the arg_attrs attribute and assume that the structure of the IR remain
// unchanged aside from additional attributes.

// The `--implicit-check-not arg_attrs` on FileCheck means that the text "arg_attrs"
// is only allowed when explicitly specified in a CHECK.


!mesh_1_tensor_1 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
!mesh_1_tensor_2 = !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>
!mesh_1_tensor_3 = !mpmd.mesh_tensor<"m1", tensor<4x16xf32>, sharding=<@mesh, [{"x"}, {?}]>>

!mesh_2_tensor_1 = !mpmd.mesh_tensor<"m2", tensor<4x16xf32>, sharding=<@mesh, [{"x"}, {?}]>>
!mesh_2_tensor_2 = !mpmd.mesh_tensor<"m2", tensor<16x8xf32>>
!mesh_2_tensor_3 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>

// CHECK-LABEL: func @mark_aliasing_for_input_with_and_without_aliasing(
func.func @mark_aliasing_for_input_with_and_without_aliasing(
  %arg0: !mesh_1_tensor_1 {tf.aliasing_output = 1 : i32},
  %arg1: !mesh_1_tensor_2,
  %arg2: !mesh_2_tensor_2 {tf.aliasing_output = 0 : i32}
) -> (
  !mesh_2_tensor_2 ,
  !mesh_2_tensor_3
) attributes {topology = #mpmd.topology<<"m1" : <["x"=2, "y"=4]>>, <"m2" : <["x"=2, "z"=3]>>>} {
  // %arg0 is marked by the user with tf.aliasing_output but it does not match
  // the op result type so it can't be aliased. Instead, it is donated.
  // CHECK: mpmd.fragment
  // CHECK-SAME: {arg_attrs = [{jax.buffer_donor = true}, {}]}
  %0 = mpmd.fragment<mesh="m1", origin=["f0"]> (%arg0, %arg1) (%arg3: tensor<4x8xf32>, %arg4: tensor<8x16xf32>) {
    %4 = "stablehlo.dot"(%arg3, %arg4) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    mpmd.return %4 : tensor<4x16xf32>
  } : (!mesh_1_tensor_1, !mesh_1_tensor_2) -> !mesh_1_tensor_3
  // %arg2 is marked by the user with tf.aliasing_output and it matches the op result type so it can be aliased.
  // CHECK: mpmd.fragment
  // CHECK-SAME: {arg_attrs = [{tf.aliasing_output = 0 : i32}]}
  %1 = mpmd.fragment<mesh="m2", origin=["f1"]> (%arg2) (%arg3: tensor<16x8xf32>) {
    %4 = stablehlo.add %arg3, %arg3 : tensor<16x8xf32>
    mpmd.return %4 : tensor<16x8xf32>
  } : (!mesh_2_tensor_2 ) -> !mesh_2_tensor_2
  %2 = mpmd.transfer %0 : (!mesh_1_tensor_3) -> !mesh_2_tensor_1
  // First input type does not match with any output type, so it is donated
  // instead of aliased.
  // CHECK: mpmd.fragment
  // CHECK-SAME: {arg_attrs = [{jax.buffer_donor = true}, {}]}
  %3 = mpmd.fragment<mesh="m2", origin=["f2"]> (%2, %1) (%arg3: tensor<4x16xf32>, %arg4: tensor<16x8xf32>) {
    %4 = "stablehlo.dot"(%arg3, %arg4) : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_2_tensor_1, !mesh_2_tensor_2 ) -> !mesh_2_tensor_3
  return %1, %3 : !mesh_2_tensor_2 , !mesh_2_tensor_3
}

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-LABEL: func.func @can_only_alias_to_one_output
func.func @can_only_alias_to_one_output(%arg0: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 1 : i32}, %arg1: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 1 : i32})
  -> (!mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // Both %arg0 and %arg1 can donate to the output but there is only one output to donate to.
  // CHECK: mpmd.fragment
  // CHECK-SAME: {arg_attrs = [{tf.aliasing_output = 0 : i32}, {jax.buffer_donor = true}]}
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg2, %arg3: tensor<4x8xf32>
    %1 = stablehlo.abs %arg3: tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  func.return %0: !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func.func @alias_to_multiple_output
func.func @alias_to_multiple_output(%arg0: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 1 : i32}, %arg1: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 0 : i32})
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // Both input can be aliased.
  // CHECK: mpmd.fragment
  // CHECK-SAME: {arg_attrs = [{tf.aliasing_output = 0 : i32}, {tf.aliasing_output = 1 : i32}]}
  %0, %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg2, %arg3: tensor<4x8xf32>
    %1 = stablehlo.abs %arg3: tensor<4x8xf32>
    mpmd.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32,!mesh_1_tensor_4_8_f32)
  func.return %0, %1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func.func @should_not_alias_offloaded_values
func.func @should_not_alias_offloaded_values(
  %arg0: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 1 : i32},
  %arg1: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 0 : i32})
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // Only the last input and the last output can be aliased, because the other
  // values are on host.
  // CHECK: mpmd.fragment
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "pinned_host"}, {tf.aliasing_output = 1 : i32}]
  %0, %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
    {arg_attrs = [{mhlo.memory_kind = "pinned_host"}, {}], res_attrs = [{mhlo.memory_kind = "pinned_host"}, {}]}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg2, %arg3: tensor<4x8xf32>
    %1 = stablehlo.abs %arg3: tensor<4x8xf32>
    mpmd.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32,!mesh_1_tensor_4_8_f32)
  func.return %0, %1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}


// CHECK-LABEL: func.func @should_not_alias_on_different_layout
func.func @should_not_alias_on_different_layout(
  %arg0: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 0 : i32},
  %arg1: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 1 : i32})
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // Only arg1 can be aliased to result1, because the layout of arg0 and
  // result0 are different.
  // CHECK: mpmd.fragment
  // CHECK-SAME: {arg_attrs = [{jax.buffer_donor = true, mhlo.layout_mode = "abc"},
  // CHECK-SAME: {tf.aliasing_output = 1 : i32}], res_attrs = [{mhlo.layout_mode = "xyz"}, {}]}
  %0, %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
    {arg_attrs = [{mhlo.layout_mode = "abc"}, {}], res_attrs = [{mhlo.layout_mode = "xyz"}, {}]}
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg2, %arg3: tensor<4x8xf32>
    %1 = stablehlo.abs %arg3: tensor<4x8xf32>
    mpmd.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32,!mesh_1_tensor_4_8_f32)
  func.return %0, %1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func.func @should_not_alias_block_argument_if_not_marked_in_arg_attribute
func.func @should_not_alias_block_argument_if_not_marked_in_arg_attribute(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // %arg0 can be aliased but the user did not mark the block argument for aliasing.
  // CHECK: mpmd.fragment
  %0, %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg2, %arg3: tensor<4x8xf32>
    %1 = stablehlo.abs %arg3: tensor<4x8xf32>
    mpmd.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32,!mesh_1_tensor_4_8_f32)
  func.return %0, %1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func.func @one_argument_used_in_multiple_fragment_should_only_alias_in_last
func.func @one_argument_used_in_multiple_fragment_should_only_alias_in_last(%arg0: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 1 : i32}, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK: mpmd.fragment
  %0, %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %0 = stablehlo.subtract %arg2, %arg3: tensor<4x8xf32>
    %1 = stablehlo.add %0, %arg3: tensor<4x8xf32>
    mpmd.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32,!mesh_1_tensor_4_8_f32)
  // CHECK: mpmd.fragment
  // CHECK-SAME: {arg_attrs = [{tf.aliasing_output = 0 : i32}, {}]}
  %2, %3 = mpmd.fragment<mesh="m1", origin=["f2"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3: tensor<4x8xf32>
    %5 = stablehlo.abs %arg3: tensor<4x8xf32>
    mpmd.return %4, %5 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32,!mesh_1_tensor_4_8_f32)
  func.return %0, %3 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func.func @transfer_op_operand_cannot_be_aliased
func.func @transfer_op_operand_cannot_be_aliased(%arg0: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 1 : i32}, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
 %0 = mpmd.transfer %arg0 : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
 %1 = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
 // %0 can be aliased with output 0 but since it is used by a transfer op so it won't be aliased.
 // CHECK: mpmd.fragment
 %4 = mpmd.fragment<mesh="m1", origin=["f1"]> (%0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %2 = stablehlo.add %arg2, %arg3: tensor<4x8xf32>
    %3 = stablehlo.abs %arg3: tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  func.return %1, %4 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func.func @transfer_result_can_be_aliased
func.func @transfer_result_can_be_aliased (%arg0: !mesh_2_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
      "topology"=#topology} {
  // CHECK: %[[TRANSFER_RESULT:.*]] = mpmd.transfer %arg0
  %param = mpmd.transfer %arg0 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK-SAME: (%[[TRANSFER_RESULT]]) {arg_attrs = [{tf.aliasing_output = 0 : i32}]}
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%param) (%arg2: tensor<4x8xf32>) {
    %add = stablehlo.add %arg2, %arg2: tensor<4x8xf32>
    mpmd.return %add : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  func.return %0 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func.func @alias_intemediates_in_chain_of_fragments
func.func @alias_intemediates_in_chain_of_fragments(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
      "topology"=#topology} {
  // CHECK: %[[FIRST_FRAGMENT_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]>
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %initial_gradient_accumulate = stablehlo.constant dense<0.000000e+00> : tensor<4x8xf32>
    %add = stablehlo.add %arg2, %initial_gradient_accumulate: tensor<4x8xf32>
    mpmd.return %add : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  // CHECK: mpmd.fragment<mesh="m1", origin=["f2"]>
  // CHECK-SAME: (%arg0, %[[FIRST_FRAGMENT_RESULT]]) {arg_attrs = [{}, {tf.aliasing_output = 0 : i32}]}
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%arg0, %0) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %add = stablehlo.add %arg2, %arg3: tensor<4x8xf32>
    mpmd.return %add : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  func.return %arg0, %1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}
