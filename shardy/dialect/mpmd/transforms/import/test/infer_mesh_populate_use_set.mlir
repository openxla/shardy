// RUN: mpmd_opt %s -mpmd-infer-mesh-populate-use-set 2>&1 | FileCheck --implicit-check-not use_set %s

!mesh_1_tensor_ui32 = !mpmd.mesh_tensor<"m1", tensor<ui32>>
!mesh_1_tensor_1_ui32 = !mpmd.mesh_tensor<"m1", tensor<1xui32>>
!mesh_1_tensor_2_ui32 = !mpmd.mesh_tensor<"m1", tensor<2xui32>>
!mesh_1_tensor_5_5_ui32 = !mpmd.mesh_tensor<"m1", tensor<5x5xui32>>
!mesh_1_tensor_4_4_f32 = !mpmd.mesh_tensor<"m1", tensor<4x4xf32>>
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_8_16_f32 = !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>
!mesh_1_tensor_4_16_f32 = !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>
!mesh_1_tensor_16_8_f32 = !mpmd.mesh_tensor<"m1", tensor<16x8xf32>>

!mesh_2_tensor_ui32 = !mpmd.mesh_tensor<"m2", tensor<ui32>>
!mesh_2_tensor_1_ui32 = !mpmd.mesh_tensor<"m2", tensor<1xui32>>
!mesh_2_tensor_2_ui32 = !mpmd.mesh_tensor<"m2", tensor<2xui32>>
!mesh_2_tensor_5_5_ui32 = !mpmd.mesh_tensor<"m2", tensor<5x5xui32>>
!mesh_2_tensor_4_4_f32 = !mpmd.mesh_tensor<"m2", tensor<4x4xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
!mesh_2_tensor_4_16_f32 = !mpmd.mesh_tensor<"m2", tensor<4x16xf32>>
!mesh_2_tensor_8_16_f32 = !mpmd.mesh_tensor<"m2", tensor<8x16xf32>>
!mesh_2_tensor_16_8_f32 = !mpmd.mesh_tensor<"m2", tensor<16x8xf32>>

!mesh_3_tensor_4_8_f32 = !mpmd.mesh_tensor<"m3", tensor<4x8xf32>>

#topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>, <"m3" : <["x"=1]>>>

// The majority of these tests verify only the use_set and assume that the DAG
// unchanged aside from additional attributes. There is one test
// `dag_unchanged_aside_from_use_set_attribute` that verifies that the DAG
// remains unchanged.
//
// The `--implicit-check-not use_set` on FileCheck means that the text "use_set"
// is only allowed when explicitly specified in a CHECK.

// CHECK-LABEL: func @single_use(%arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1"["abc"]>})
func.func @single_use(%arg0: tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1"["abc"]>
  %1 = stablehlo.add %arg0, %0 : tensor<4x8xf32> // CHECK-NEXT: use_set = {{.*}}"m1"["abc"]>
  // note: ignores origin
  %2 = mpmd.assign {origin = "abc"} %1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32   // CHECK-NEXT: use_set = {{.*}}"m1"["abc"]>
  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @multi_use_same_mesh_same_origin(%arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1"["1"]>})
func.func @multi_use_same_mesh_same_origin(%arg0: tensor<4x8xf32>)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32> // CHECK-NEXT: use_set = {{.*}}"m1"["1"]>
  %1 = stablehlo.add %arg0, %0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1"["1"]>
  %2 = mpmd.assign {origin = "1"} %1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK-NEXT: use_set = {{.*}}"m1"["1"]>

  %3 = stablehlo.add %arg0, %1 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1"["1"]>
  %4 = mpmd.assign {origin = "1"} %3 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK-NEXT: use_set = {{.*}}"m1"["1"]>
  func.return %2, %4 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @multi_use_same_mesh_different_origin(%arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1"["2", "1"]>})
func.func @multi_use_same_mesh_different_origin(%arg0: tensor<4x8xf32>)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %1 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1"["2", "1"]>
  %2 = mpmd.assign {origin = "1"} %1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK-NEXT: use_set = {{.*}}"m1"["1"]>

  %3 = stablehlo.add %arg0, %1 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1"["2"]>
  %4 = mpmd.assign {origin = "2"} %3 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK-NEXT: use_set = {{.*}}"m1"["2"]>
  func.return %2, %4 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}


// Regression test to check that we keep the arg's use_set and add to it, rather
// than replacing it. This is relevant when propagating into MPMD callees:
// the callee arg may have use_set populated from the caller
// (e.g. arg -> return -> caller use), and we need to keep that use_set and
// add to it, rather than erasing it.
// CHECK-LABEL: func @arg_use_set_not_overridden(%arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2", "m1">})
func.func @arg_use_set_not_overridden(%arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2">})
  -> (!mesh_1_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %2 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK-NEXT: use_set = {{.*}}"m1">
  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @multi_use_multi_mesh(%arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2", "m1">})
func.func @multi_use_multi_mesh(%arg0: tensor<4x8xf32>)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m2", "m1">
  %1 = stablehlo.add %arg0, %0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m2", "m1">
  %2 = mpmd.assign %1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32 // CHECK-NEXT: use_set = {{.*}}"m1">

  %3 = stablehlo.add %arg0, %1 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m2">
  %4 = mpmd.assign %3 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32  // CHECK-NEXT: use_set = {{.*}}"m2">
  func.return %2, %4 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @region_op_region_is_skipped(%arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1">})
func.func @region_op_region_is_skipped(%arg0: tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant dense<1> : tensor<i32>  // CHECK-NEXT: use_set = {{.*}}"m1">
  %5:2 = stablehlo.while(%iterArg_0 = %0, %iterArg_1 = %arg0) : tensor<i32>, tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1">
   cond {
    %6 = "stablehlo.compare"(%iterArg_0, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%6) : (tensor<i1>) -> ()
  } do {
    %8 = stablehlo.add %iterArg_1, %iterArg_1 : tensor<4x8xf32>
    "stablehlo.return"(%iterArg_0, %8) : (tensor<i32>, tensor<4x8xf32>) -> ()
  }
  %6 = mpmd.assign %5#1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m1">

  func.return %6 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @op_and_value_used_as_free_variable_in_region_op(
// CHECK-SAME:    %arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1">},
// CHECK-SAME:    %arg1: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1">})
func.func @op_and_value_used_as_free_variable_in_region_op(
  %arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant dense<1> : tensor<i32>  // CHECK-NEXT: use_set = {{.*}}"m1">
  %1 = stablehlo.constant dense<2.0> : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1">
  %5:2 = stablehlo.while(%iterArg_0 = %0, %iterArg_1 = %arg0) : tensor<i32>, tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1">
   cond {
    %6 = "stablehlo.compare"(%iterArg_0, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%6) : (tensor<i1>) -> ()
  } do {
    %8 = stablehlo.add %1, %arg1 : tensor<4x8xf32>
    "stablehlo.return"(%iterArg_0, %8) : (tensor<i32>, tensor<4x8xf32>) -> ()
  }
  %6 = mpmd.assign %5#1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m1">

  func.return %6 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @no_naked_ops(%arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2", "m1">})
func.func @no_naked_ops(%arg0: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %1 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK-NEXT: use_set = {{.*}}"m1">
  %2 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32  // CHECK-NEXT: use_set = {{.*}}"m2">
  func.return %1, %2 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @unassign_assign
func.func @unassign_assign(%arg0: !mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  // Since unassign is an MPMD op, it will not get a use_set.
  // The only MPMD op that gets use_sets are assign_ops.
  %1 = mpmd.unassign %arg0 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}use_set = {{.*}}"m1">
  %2 = mpmd.assign %1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m1">
  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @assign_transfer_unassign(%arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2">})
func.func @assign_transfer_unassign(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %1 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32  // CHECK-NEXT: use_set = {{.*}}"m2">
  %2 = mpmd.transfer %1 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %3 = mpmd.unassign %2 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  func.return %3 : tensor<4x8xf32>
}

// CHECK-LABEL: func @function_without_assign_op_is_noop
func.func @function_without_assign_op_is_noop(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @ops_in_fragments_need_no_analysis
func.func @ops_in_fragments_need_no_analysis(%arg0: !mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %f = mpmd.fragment<mesh="m2", origin=["foo"]> (%arg0) (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %0 : tensor<4x8xf32>
  } : (!mesh_2_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  %3 = mpmd.unassign %f : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: assign {{.*}}use_set = {{.*}}"m1">
  %4 = mpmd.assign %3 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m1">
  func.return %4 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @assign_outside_call_op
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2">}
func.func @assign_outside_call_op(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1">

  %2:2 = mpmd.call @assign_outside_call_op_f(%0, %arg1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %3 = mpmd.assign %2#0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m1">
  %4 = mpmd.assign %2#1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m2">
  func.return %3, %4 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}
// CHECK:      func.func private @assign_outside_call_op_f
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.use_set = {{[^,-]*}}"m1">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.use_set = {{[^,-]*}}"m2">}
// CHECK-SAME: -> (tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1">}, tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2">})
func.func private @assign_outside_call_op_f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1">

  // Verify that use_set propagates through call_op to args which
  // are returned directly.
  func.return %0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
}


// CHECK-LABEL: func @assign_outside_call_op_multiple_calls
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1", "m2">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2">}
func.func @assign_outside_call_op_multiple_calls(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  // Note that the first operand of the call_op %0 and %32 have different
  // use_sets. Thus the use_sets aren't exactly correct since it should both be
  // {m1,m2}. We will throw a validation error in another pass to handle this,
  // as the edges out of a call_op can't be assigned to multiple meshes and so
  // they should have a single-element use_set.
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1", "m2">
  %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m2">

  %2:2 = mpmd.call @assign_outside_call_op_multiple_calls_f(%0, %1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %3 = mpmd.assign %2#0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m2">
  %31 = mpmd.unassign %3 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}use_set = {{.*}}"m1">
  %32 = stablehlo.add %31, %31 : tensor<4x8xf32>  // CHECK: add {{.*}}use_set = {{.*}}"m1">

  %4:2 = mpmd.call @assign_outside_call_op_multiple_calls_f(%32, %2#1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %6 = mpmd.assign %4#0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m1">
  %7 = mpmd.assign %4#1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m2">
  func.return %6, %7 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}
// CHECK: func.func private @assign_outside_call_op_multiple_calls_f
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.use_set = {{[^,-]*}}"m1", "m2">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.use_set = {{[^,-]*}}"m2">}
// CHECK-SAME: -> (tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1", "m2">}, tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2">})
func.func private @assign_outside_call_op_multiple_calls_f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1", "m2">
  %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m2">

  func.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @assign_op_is_in_call_body
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2">}
func.func @assign_op_is_in_call_body(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1">
  %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m2">

  %2:2 = mpmd.call @assign_op_is_in_call_body_f(%0, %1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  func.return %2#0, %2#1 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}
// CHECK: func.func private @assign_op_is_in_call_body_f
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2">}
func.func private @assign_op_is_in_call_body_f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1">
  %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m2">

  %3 = mpmd.assign %0: (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m1">
  %4 = mpmd.assign %1: (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m2">
  func.return %3, %4 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @assign_op_is_in_call_body_multiple_calls
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2">}
func.func @assign_op_is_in_call_body_multiple_calls(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1">
  %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m2">

  %2:2 = mpmd.call @assign_op_is_in_call_body_multiple_calls_f(%0, %1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  %3 = mpmd.unassign %2#0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}use_set = {{.*}}"m1">
  %4 = mpmd.unassign %2#1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}use_set = {{.*}}"m2">
  %5 = stablehlo.add %3, %3 : tensor<4x8xf32>  // CHECK: add {{.*}}use_set = {{.*}}"m1">
  %6 = stablehlo.add %4, %4 : tensor<4x8xf32>  // CHECK: add {{.*}}use_set = {{.*}}"m2">

  %7:2 = mpmd.call @assign_op_is_in_call_body_multiple_calls_f(%5, %6) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)

  func.return %7#0, %7#1 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}
// CHECK: func.func private @assign_op_is_in_call_body_multiple_calls_f
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2">}
func.func private @assign_op_is_in_call_body_multiple_calls_f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m1">
  %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>  // CHECK-NEXT: use_set = {{.*}}"m2">

  %3 = mpmd.assign %0: (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m1">
  %4 = mpmd.assign %1: (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32  // CHECK: assign {{.*}}use_set = {{.*}}"m2">
  func.return %3, %4 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @call_op_needs_multiple_iterations_to_converge
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m3", "m2", "m1">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1", "m3", "m2">}
// CHECK-SAME:   arg2: tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2", "m1", "m3">}
func.func @call_op_needs_multiple_iterations_to_converge(
      %arg0: tensor<4x8xf32>,
      %arg1: tensor<4x8xf32>,
      %arg2: tensor<4x8xf32>) -> (
  !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_3_tensor_4_8_f32)
  attributes {topology=#topology}
{
  %1:3 = mpmd.call @call_op_needs_multiple_iterations_to_converge_f(%arg0, %arg1, %arg2) : (
    tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  ) -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)

  %3:3 = mpmd.call @call_op_needs_multiple_iterations_to_converge_f(%1#0, %1#1, %1#2) : (
    tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  ) -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  %4 = mpmd.assign %3#0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32 // CHECK: assign {{.*}}use_set = {{.*}}"m1">
  %5 = mpmd.assign %3#1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32 // CHECK: assign {{.*}}use_set = {{.*}}"m2">
  %6 = mpmd.assign %3#2 : (tensor<4x8xf32>) -> !mesh_3_tensor_4_8_f32 // CHECK: assign {{.*}}use_set = {{.*}}"m3">


  return %4, %5, %6 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_3_tensor_4_8_f32
}

// Each of the calls adds one additional use_set mesh. So two calls would
// give two meshes. The existence of three meshes means that we propagate
// more times than there are calls
// CHECK-LABEL: func.func private @call_op_needs_multiple_iterations_to_converge_f(
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.use_set = {{[^,-]*}}"m3", "m2", "m1">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.use_set = {{[^,-]*}}"m1", "m3", "m2">}
// CHECK-SAME:   arg2: tensor<4x8xf32> {mpmd.use_set = {{[^,-]*}}"m2", "m1", "m3">}
// CHECK-SAME: -> (tensor<4x8xf32> {mpmd.use_set = {{.*}}"m1", "m3", "m2">},
// CHECK-SAME:     tensor<4x8xf32> {mpmd.use_set = {{.*}}"m2", "m1", "m3">},
// CHECK-SAME:     tensor<4x8xf32> {mpmd.use_set = {{.*}}"m3", "m2", "m1">})
func.func private @call_op_needs_multiple_iterations_to_converge_f(
  %arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>
) -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) attributes
{topology = #topology} {
  return %arg1, %arg2, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
}



// CHECK-LABEL: func @dag_unchanged_aside_from_use_set_attribute
// CHECK-SAME:     %arg0: tensor<4x8xf32> {mpmd.use_set = {{[^,-]*}}"m1">}
// CHECK-SAME:     %arg1: tensor<8x16xf32> {mpmd.use_set = {{[^,-]*}}"m1">}
// CHECK-SAME:     %arg2: tensor<16x8xf32> {mpmd.use_set = {{[^,-]*}}"m2">}
func.func @dag_unchanged_aside_from_use_set_attribute(
  %arg0: tensor<4x8xf32>,
  %arg1: tensor<8x16xf32>,
  %arg2: tensor<16x8xf32>,
  %arg3: tensor<4x16xf32>,
  %arg4: tensor<16x8xf32>)
  -> (tensor<4x16xf32>, tensor<4x8xf32>, tensor<16x8xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
  // CHECK-NEXT: %0 = stablehlo.add %arg2, %arg4
  %0 = stablehlo.add %arg2, %arg4 : tensor<16x8xf32>

  // CHECK-NEXT: %1 = mpmd.assign {mpmd.use_set = {{.*}}"m1">} %arg0
  %1 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  // CHECK-NEXT: %2 = mpmd.assign {mpmd.use_set = {{.*}}"m1">} %arg1
  %2 = mpmd.assign %arg1 : (tensor<8x16xf32>) -> !mesh_1_tensor_8_16_f32

  // CHECK-NEXT: %3 = mpmd.fragment<mesh="m1", origin=["f1"]> (%1, %2)
  // CHECK-SAME:  (%arg5: tensor<4x8xf32>, %arg6: tensor<8x16xf32>) {
  %3 = mpmd.fragment<mesh="m1", origin=["f1"]> (%1, %2)
    (%arg5: tensor<4x8xf32>, %arg6: tensor<8x16xf32>) {

    // CHECK-NEXT: %12 = stablehlo.dot %arg5, %arg6 : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %12 = "stablehlo.dot"(%arg5, %arg6) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    // CHECK-NEXT: mpmd.return %12 : tensor<4x16xf32>
    mpmd.return %12 : tensor<4x16xf32>
    // CHECK-NEXT: }
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_8_16_f32) -> (!mesh_1_tensor_4_16_f32)

  // CHECK-NEXT: %4 = mpmd.transfer %3
  %4 = mpmd.transfer %3 : (!mesh_1_tensor_4_16_f32) -> !mesh_2_tensor_4_16_f32

  // CHECK-NEXT: %5 = stablehlo.add %arg2, %arg2 {mpmd.use_set = {{.*}}"m2">}
  %5 = stablehlo.add %arg2, %arg2 : tensor<16x8xf32>

  // CHECK-NEXT: %6 = mpmd.assign {mpmd.use_set = {{.*}}"m2">} %5
  %6 = mpmd.assign %5 : (tensor<16x8xf32>) -> !mesh_2_tensor_16_8_f32

  // CHECK-NEXT: %7 = mpmd.fragment<mesh="m2", origin=["f2"]> (%4, %6)
  // CHECK-SAME:   (%arg5: tensor<4x16xf32>, %arg6: tensor<16x8xf32>) {
  %7 = mpmd.fragment<mesh="m2", origin=["f2"]> (%4, %6)
    (%arg5: tensor<4x16xf32>, %arg6: tensor<16x8xf32>) {
    // CHECK-NEXT: %12 = stablehlo.dot %arg5, %arg6 : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
    %12 = "stablehlo.dot"(%arg5, %arg6) : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
    // CHECK-NEXT: mpmd.return %12 : tensor<4x8xf32>
    mpmd.return %12 : tensor<4x8xf32>
  // CHECK-NEXT: }
  } : (!mesh_2_tensor_4_16_f32, !mesh_2_tensor_16_8_f32) -> !mesh_2_tensor_4_8_f32

  // CHECK-NEXT: %8 = mpmd.unassign %3
  %8 = mpmd.unassign %3 : (!mesh_1_tensor_4_16_f32) -> tensor<4x16xf32>

  // CHECK-NEXT: %9 = stablehlo.add %8, %arg3 : tensor<4x16xf32>
  %9 = stablehlo.add %8, %arg3 : tensor<4x16xf32>
  // CHECK-NEXT: %10 = stablehlo.add %9, %9 : tensor<4x16xf32>
  %10 = stablehlo.add %9, %9 : tensor<4x16xf32>

  // CHECK-NEXT: %11 = mpmd.unassign %7
  %11 = mpmd.unassign %7 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>

  // CHECK-NEXT: return %10, %11, %0 : tensor<4x16xf32>, tensor<4x8xf32>, tensor<16x8xf32>
  func.return %10, %11, %0 : tensor<4x16xf32>, tensor<4x8xf32>, tensor<16x8xf32>
}

// CHECK-LABEL: func @func_with_no_topology_no_propagation_because_skipped
func.func @func_with_no_topology_no_propagation_because_skipped(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %0 = stablehlo.add
  // CHECK-NEXT: %1 = stablehlo.add  {{.*}}use_set = {{.*}}"mesh">
  // CHECK-NEXT: return %1
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  %1 = stablehlo.add %0, %0 {mpmd.use_set = #mpmd.meshes_with_origins<"mesh">} : tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @broadcast_is_a_barrier_while_populating_use_set
func.func @broadcast_is_a_barrier_while_populating_use_set(%arg0: tensor<4x8xf32>) -> (
  !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>}
{
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: %[[BCAST:.*]] = mpmd.broadcast {mpmd.use_set = {{.*}}"m2", "m1">} %[[ADD]] : tensor<4x8xf32>
  %bcast = mpmd.broadcast %add : tensor<4x8xf32>
  // CHECK-NEXT: mpmd.assign {mpmd.use_set = {{.*}}"m1">}
  // CHECK-NEXT: mpmd.assign {mpmd.use_set = {{.*}}"m2">}
  %a1 = mpmd.assign %bcast : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %a2 = mpmd.assign %bcast : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  func.return %a1, %a2 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
}

// CHECK-LABEL: func @reduce_is_a_barrier_while_populating_use_set
func.func @reduce_is_a_barrier_while_populating_use_set(%arg0: tensor<4x8xf32>) -> (
  !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>}
{
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // CHECK-NEXT: %[[BCAST:.*]] = mpmd.reduce<add> {mpmd.use_set = {{.*}}"m2", "m1">} %[[ADD]], %[[ADD]]
  %reduce = mpmd.reduce<add> %add, %add : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: mpmd.assign {mpmd.use_set = {{.*}}"m1">}
  // CHECK-NEXT: mpmd.assign {mpmd.use_set = {{.*}}"m2">}
  %a1 = mpmd.assign %reduce : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %a2 = mpmd.assign %reduce : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  func.return %a1, %a2 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
}

// CHECK-LABEL: func @fori_loop(
// CHECK-SAME: %arg0: tensor<ui32> {mpmd.use_set = {{.*}}"m1">}
func.func @fori_loop(%arg0: tensor<ui32>) -> !mesh_1_tensor_ui32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<ui32>  // CHECK-NEXT: use_set = {{.*}}"m1">
  %1 = mpmd.for (%0) {iterations = 3 : ui32, unroll_factor = 3 : ui32} (%arg1: tensor<ui32>, %index: tensor<ui32>) {  // CHECK-NEXT: for {{.*}}arg_attrs = [{mpmd.use_set = {{.*}}"m1">}, {mpmd.use_set = {{.*}}"m1">}]
    %3 = stablehlo.constant dense<1> : tensor<ui32>  // CHECK-NEXT: use_set = {{.*}}"m1">
    %4 = stablehlo.add %arg1, %3 : tensor<ui32> // CHECK-NEXT: use_set = {{.*}}"m1">
    %5 = mpmd.call @fori_loop_f(%4, %index) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    mpmd.return %5 : tensor<ui32>
  } : tensor<ui32>
  %2 = mpmd.assign %1 : (tensor<ui32>) -> !mesh_1_tensor_ui32  // CHECK: assign {{.*}}use_set = {{.*}}"m1">
  return %2 : !mesh_1_tensor_ui32
}

// CHECK-LABEL: func private @fori_loop_f(
// CHECK-SAME: %arg0: tensor<ui32> {mpmd.use_set = {{[^,-]*}}"m1">}
// CHECK-SAME: %arg1: tensor<ui32> {mpmd.use_set = {{[^,-]*}}"m1">}
// CHECK-SAME: -> (tensor<ui32> {mpmd.use_set = {{.*}}"m1">})
func.func private @fori_loop_f(%arg0: tensor<ui32>, %arg1: tensor<ui32>) -> tensor<ui32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>>} {
  %0 = stablehlo.add %arg0, %arg1 : tensor<ui32>  // CHECK-NEXT: use_set = {{.*}}"m1">
  return %0 : tensor<ui32>
}
