// RUN: mpmd_opt %s -mpmd-infer-mesh-populate-src-set 2>&1 | FileCheck --implicit-check-not src_set %s

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

// The majority of these tests verify only the src_set and assume that the DAG
// unchanged aside from additional attributes. There is one test
// `dag_unchanged_aside_from_src_set_attribute` that verifies that the DAG
// remains unchanged.
//
// The `--implicit-check-not src_set` on FileCheck means that the text "src_set"
// is only allowed when explicitly specified in a CHECK.

// CHECK-LABEL: func @single_source(%arg0: !mpmd.mesh_tensor<"m1"
func.func @single_source(%arg0: !mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  %2 = stablehlo.add %1, %0 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  func.return %2 : tensor<4x8xf32>
}

// CHECK-LABEL: func @multi_source_same_mesh_same_origin(%arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1"
func.func @multi_source_same_mesh_same_origin(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> (tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %1 = mpmd.unassign {origin = "1"} %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1"["1"]>
  %2 = mpmd.unassign {origin = "1"} %arg1 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1"["1"]>
  %3 = stablehlo.add %1, %2 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1"["1"]>
  %4 = stablehlo.add %0, %3 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1"["1"]>
  func.return %4 : tensor<4x8xf32>
}

// CHECK-LABEL: func @multi_source_same_mesh_different_origin(%arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1"
func.func @multi_source_same_mesh_different_origin(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> (tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %1 = mpmd.unassign {origin = "1"} %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1"["1"]>
  %2 = mpmd.unassign {origin = "2"} %arg1 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1"["2"]>
  %3 = stablehlo.add %1, %2 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1"["1", "2"]>
  %4 = stablehlo.add %0, %3 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1"["1", "2"]>
  func.return %4 : tensor<4x8xf32>
}

// CHECK-LABEL: func @multi_source_multi_mesh_no_common(%arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m2"
func.func @multi_source_multi_mesh_no_common(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> (tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  %2 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  // The src_sets for these ops are empty since `intersect({m1}, {m2}) = {}`,
  // but they are added so they are detected as a reduce and get
  // `src_set = union(operand_src_sets)`
  %3 = stablehlo.add %1, %2 : tensor<4x8xf32> // CHECK-NEXT: src_set = {{.*}}"m1", "m2">
  %4 = stablehlo.add %0, %3 : tensor<4x8xf32> // CHECK-NEXT: src_set = {{.*}}"m1", "m2">
  func.return %4 : tensor<4x8xf32>
}

// CHECK-LABEL: func @multi_source_multi_mesh_common(%arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m2"
func.func @multi_source_multi_mesh_common(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> (tensor<4x8xf32>, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // Note the constant has no src_set, so it defaults to all meshes and we
  // can verify that the intersection works when the src_set is absent.
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1", "m2"["transfer"]>
  %2 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  %3 = stablehlo.add %1, %2 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2"["transfer"]>
  %4 = stablehlo.add %0, %3 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2"["transfer"]>
  %5 = mpmd.transfer %arg0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  func.return %4, %5 : tensor<4x8xf32>, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @region_op_region_is_skipped(
func.func @region_op_region_is_skipped(%arg0: !mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant {mpmd.src_set = #mpmd.meshes_with_origins<"m1">} dense<1> : tensor<i32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %5:2 = stablehlo.while(%iterArg_0 = %0, %iterArg_1 = %1) : tensor<i32>, tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
   cond {
    %6 = "stablehlo.compare"(%iterArg_0, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%6) : (tensor<i1>) -> ()
  } do {
    %8 = stablehlo.add %iterArg_1, %iterArg_1 : tensor<4x8xf32>
    "stablehlo.return"(%iterArg_0, %8) : (tensor<i32>, tensor<4x8xf32>) -> ()
  }

  func.return %5#1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @func_arg_initialization_and_propagation(
// CHECK-SAME:    %arg0: tensor<4x8xf32> {mpmd.src_set = {{.*}}"m2"["inferred_in"], "m1"["inferred_in"]>, mpmd.use_set = {{.*}}"m2"["stage0", "stage1"], "m1">}
// CHECK-SAME:    %arg1: tensor<4x8xf32> {mpmd.src_set = {{.*}}"m1"["inferred_in"]>, mpmd.use_set = {{.*}}"m1">})
func.func @func_arg_initialization_and_propagation(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2"["stage0", "stage1"], "m1">},
  %arg1: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">}) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %1 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2"["inferred_in"], "m1"["inferred_in"]>
  %2 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2"["inferred_in"], "m1"["inferred_in"]>
  %3 = stablehlo.add %1, %2  : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2"["inferred_in"], "m1"["inferred_in"]>
  func.return %3 : tensor<4x8xf32>
}

// CHECK-LABEL: func @unassign_of_func_arg(
func.func @unassign_of_func_arg(%arg0: !mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1", "m2"["inferred_in"], "m3"["inferred_in"]>
  %2 = stablehlo.add %1, %1 {mpmd.use_set = #mpmd.meshes_with_origins<"m3">} : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1", "m2"["inferred_in"], "m3"["inferred_in"]>
  %3 = stablehlo.add %1, %1 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1", "m2"["inferred_in"], "m3"["inferred_in"]>

  func.return %2 : tensor<4x8xf32>
}


// CHECK-LABEL: func @no_naked_ops
func.func @no_naked_ops(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32) -> (tensor<4x8xf32>,  tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %2 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>   // CHECK-NEXT: src_set = {{.*}}"m2">
  func.return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @unassign_assign
func.func @unassign_assign(%arg0: !mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %1 = mpmd.unassign %arg0 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  %2 = mpmd.assign %1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32 // CHECK-NEXT: src_set = {{.*}}"m2">
  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @assign_transfer_unassign
func.func @assign_transfer_unassign(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %1 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  %2 = mpmd.transfer %1 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %3 = mpmd.unassign %2 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  func.return %3 : tensor<4x8xf32>
}

// CHECK-LABEL: func @function_without_unassign_op_is_noop
func.func @function_without_unassign_op_is_noop(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @ops_in_fragments_need_no_analysis
func.func @ops_in_fragments_need_no_analysis(%arg0: !mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %2 = mpmd.assign %1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32  // CHECK-NEXT: src_set = {{.*}}"m1">
  %f = mpmd.fragment<mesh="m2", origin=["foo"]> (%2) (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %0 : tensor<4x8xf32>
  } : (!mesh_2_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  func.return %f : !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @unassign_outside_call_op
func.func @unassign_outside_call_op(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %1 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  %2:2 = mpmd.call @unassign_outside_call_op_f(%0, %1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %3 = stablehlo.add %2#0, %2#0 : tensor<4x8xf32>  // CHECK: stablehlo.add {{.*}}src_set = {{.*}}"m1">
  %4 = stablehlo.add %2#1, %2#1 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  func.return %3, %4 : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK:      func.func private @unassign_outside_call_op_f
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.src_set = {{.*}}"m1">}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.src_set = {{.*}}"m2">}
func.func private @unassign_outside_call_op_f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  func.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}


// CHECK-LABEL: func @unassign_outside_call_op_multiple_calls
func.func @unassign_outside_call_op_multiple_calls(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %1 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  %2:2 = mpmd.call @unassign_outside_call_op_multiple_calls_f(%0, %1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %3 = mpmd.assign %2#0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32   // CHECK: assign {{.*}}src_set = {{.*}}"m1">
  %31 = mpmd.unassign %3 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m2">

  %4:2 = mpmd.call @unassign_outside_call_op_multiple_calls_f(%31, %2#1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %5 = stablehlo.add %4#0, %4#0 : tensor<4x8xf32>  // CHECK: stablehlo.add {{.*}}src_set = {{.*}}<>
  %6 = stablehlo.add %4#1, %4#1 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  func.return %5, %6 : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK:      func.func private @unassign_outside_call_op_multiple_calls_f
// CHECK-SAME:   arg0: tensor<4x8xf32> {mpmd.src_set = {{[^,]*}}<>}
// CHECK-SAME:   arg1: tensor<4x8xf32> {mpmd.src_set = {{.*}}"m2">}
func.func private @unassign_outside_call_op_multiple_calls_f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}<>
  %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  func.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}


// CHECK-LABEL: func @unassign_op_is_in_call_body
func.func @unassign_op_is_in_call_body(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %2:2 = mpmd.call @unassign_op_is_in_call_body_f(%arg0, %arg1) : (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %3 = stablehlo.add %2#0, %2#0 : tensor<4x8xf32>  // CHECK: stablehlo.add {{.*}}src_set = {{.*}}"m1">
  %4 = stablehlo.add %2#1, %2#1 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  func.return %3, %4 : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK:      func.func private @unassign_op_is_in_call_body_f
func.func private @unassign_op_is_in_call_body_f(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %1 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  %2 = stablehlo.add %0, %0 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %3 = stablehlo.add %1, %1 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  func.return %2, %3 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @unassign_op_is_in_call_body_multiple_calls
func.func @unassign_op_is_in_call_body_multiple_calls(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %2:2 = mpmd.call @unassign_op_is_in_call_body_multiple_calls_f(%arg0, %arg1) : (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %3 = stablehlo.add %2#0, %2#0 : tensor<4x8xf32>  // CHECK: stablehlo.add {{.*}}src_set = {{.*}}"m1">
  %4 = stablehlo.add %2#1, %2#1 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  %31 = mpmd.assign %3 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32  // CHECK-NEXT: src_set = {{.*}}"m1">
  %41 = mpmd.assign %4 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32  // CHECK-NEXT: src_set = {{.*}}"m2">
  %5:2 = mpmd.call @unassign_op_is_in_call_body_multiple_calls_f(%31, %41) : (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %6 = stablehlo.add %5#0, %5#0 : tensor<4x8xf32>  // CHECK: stablehlo.add {{.*}}src_set = {{.*}}"m1">
  %7 = stablehlo.add %5#1, %5#1 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  func.return %6, %7 : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK:      func.func private @unassign_op_is_in_call_body_multiple_calls_f
func.func private @unassign_op_is_in_call_body_multiple_calls_f(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %1 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  %2 = stablehlo.add %0, %0 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %3 = stablehlo.add %1, %1 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  func.return %2, %3 : tensor<4x8xf32>, tensor<4x8xf32>
}

// Regression test, to verify that we don't override the src_set of the callee
// arg. It should correspond the the callee operand, not to the use_set of the
// callee.
// CHECK-LABEL: func @callee_arg_src_set_is_from_operand
func.func @callee_arg_src_set_is_from_operand(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %arg0_a = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %arg1_a = mpmd.assign %arg1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  %0 = mpmd.unassign %arg0_a : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK: mpmd.unassign {{.*}}src_set = {{.*}}"m1">
  %1 = mpmd.unassign %arg1_a : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  // CHECK-NEXT: mpmd.call @callee_arg_src_set_is_from_operand_f
  %2:2 = mpmd.call @callee_arg_src_set_is_from_operand_f(%0, %1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %2#0, %2#1 : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK:      func.func private @callee_arg_src_set_is_from_operand_f
// CHECK-SAME:   arg0: tensor<4x8xf32> {{.*}}mpmd.src_set = {{.*}}"m1">
// CHECK-SAME:   arg1: tensor<4x8xf32> {{.*}}mpmd.src_set = {{.*}}"m2">
func.func private @callee_arg_src_set_is_from_operand_f(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m3">},
  %arg1: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m3">}) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m1">
  %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">

  func.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @dag_unchanged_aside_from_src_set_attribute
func.func @dag_unchanged_aside_from_src_set_attribute(
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

  // CHECK-NEXT: %1 = mpmd.assign %arg0
  %1 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  // CHECK-NEXT: %2 = mpmd.assign %arg1
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

  // CHECK-NEXT: %5 = stablehlo.add %arg2, %arg2
  %5 = stablehlo.add %arg2, %arg2 : tensor<16x8xf32>

  // CHECK-NEXT: %6 = mpmd.assign %5
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

  // CHECK-NEXT: %8 = mpmd.unassign {mpmd.src_set = {{.*}}"m1", "m2"["transfer"]>} %3
  %8 = mpmd.unassign %3 : (!mesh_1_tensor_4_16_f32) -> tensor<4x16xf32>

  // CHECK-NEXT: %9 = stablehlo.add %8, %arg3 {mpmd.src_set = {{.*}}"m1", "m2"["transfer"]>} : tensor<4x16xf32>
  %9 = stablehlo.add %8, %arg3 : tensor<4x16xf32>
  // CHECK-NEXT: %10 = stablehlo.add %9, %9 {mpmd.src_set = {{.*}}"m1", "m2"["transfer"]>} : tensor<4x16xf32>
  %10 = stablehlo.add %9, %9 : tensor<4x16xf32>

  // CHECK-NEXT: %11 = mpmd.unassign {mpmd.src_set = {{.*}}"m2">} %7
  %11 = mpmd.unassign %7 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>

  // CHECK-NEXT: return %10, %11, %0 : tensor<4x16xf32>, tensor<4x8xf32>, tensor<16x8xf32>
  func.return %10, %11, %0 : tensor<4x16xf32>, tensor<4x8xf32>, tensor<16x8xf32>
}

// CHECK-LABEL: func @func_with_no_topology_no_propagation_because_skipped
func.func @func_with_no_topology_no_propagation_because_skipped(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %0 = stablehlo.add {{.*}}src_set = {{.*}}"mesh">
  // CHECK-NEXT: %1 = stablehlo.add
  // CHECK-NEXT: return %1
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"mesh">} : tensor<4x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @broadcast_src_set
func.func @broadcast_src_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>  attributes {
  "topology"=#mpmd.topology<
    <"mesh": <["x"=2]>>
  >}
{
  // The src_set of a broadcast includes all meshes, i.e., it's undefined.
  // CHECK-NEXT: mpmd.broadcast {mpmd.use_set = {{.*}}"mesh">}
  %0 = mpmd.broadcast {mpmd.use_set = #mpmd.meshes_with_origins<"mesh">} %arg0 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @fori_loop(%arg0: !mpmd.mesh_tensor<"m1"
func.func @fori_loop(%arg0: !mesh_1_tensor_ui32) -> tensor<ui32>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  %0 = mpmd.unassign %arg0 : (!mesh_1_tensor_ui32) -> tensor<ui32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  %1 = mpmd.for (%0) {iterations = 3 : ui32, unroll_factor = 3 : ui32} (%arg1: tensor<ui32>, %index: tensor<ui32>) {  // CHECK-NEXT: for {{.*}}arg_attrs = [{mpmd.src_set = {{.*}}"m1">}, {}]
    %2 = stablehlo.constant dense<1> : tensor<ui32>
    %3 = stablehlo.add %arg1, %2 : tensor<ui32>  // CHECK: src_set = {{.*}}"m1">
    %4 = stablehlo.add %3, %index : tensor<ui32>  // CHECK-NEXT: src_set = {{.*}}"m1">
    mpmd.return %4 : tensor<ui32>
  } : tensor<ui32>
  %2 = stablehlo.add %1, %1 : tensor<ui32>  // CHECK: src_set = {{.*}}"m1">
  return %2 : tensor<ui32>
}
