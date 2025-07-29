// RUN: mpmd_opt %s -mpmd-infer-mesh-assign-mesh-func-leaves='infer-transfers=true' 2>&1 | FileCheck --implicit-check-not use_set %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @unused_input(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
func.func @unused_input(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @output_without_src_set_or_use_set(%arg0: tensor<4x8xf32>)
func.func @output_without_src_set_or_use_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // m1 is picked for the output because we pick the first mesh in the topology.
  // CHECK:      %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %[[ADD]] {{.*}}mesh_tensor<"m1"
  // CHECK-NEXT: return %[[ASSIGN]]

  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @output_with_src_set(%arg0: tensor<4x8xf32>)
func.func @output_with_src_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %[[ADD]] {{.*}}mesh_tensor<"m3"
  // CHECK-NEXT: return %[[ASSIGN]]

  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @output_with_src_set_multiple(%arg0: tensor<4x8xf32>)
func.func @output_with_src_set_multiple(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // Both m2 and m3 are valid for the output, but m2 is picked because we pick
  // the first mesh in the src_set since the use_set is empty.
  // CHECK:      %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %[[ADD]] {{.*}}mesh_tensor<"m2"
  // CHECK-NEXT: return %[[ASSIGN]]

  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @output_with_use_set(%arg0: tensor<4x8xf32>)
func.func @output_with_use_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %[[ADD]] {{.*}}mesh_tensor<"m3"
  // CHECK-NEXT: return %[[ASSIGN]]

  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @output_with_use_set_multiple(%arg0: tensor<4x8xf32>)
func.func @output_with_use_set_multiple(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %[[ADD]] {{.*}}mesh_tensor<"m2"
  // CHECK-NEXT: return %[[ASSIGN]]

  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m2", "m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @output_with_use_set_contained_in_src_set(%arg0: tensor<4x8xf32>
func.func @output_with_use_set_contained_in_src_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %[[ADD]] {{.*}}mesh_tensor<"m3"
  // CHECK-NEXT: return %[[ASSIGN]]

  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @output_with_use_set_equal_to_src_set(%arg0: tensor<4x8xf32>)
func.func @output_with_use_set_equal_to_src_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %[[ADD]] {{.*}}mesh_tensor<"m2"
  // CHECK-NEXT: return %[[ASSIGN]]

  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m3", "m2">, mpmd.use_set = #mpmd.meshes_with_origins<"m3", "m2">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @output_direct_from_input_no_src_set_or_use_set
func.func @output_direct_from_input_no_src_set_or_use_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %arg0 {{.*}}mesh_tensor<"m1"
  // CHECK-NEXT: return %[[ASSIGN]]

  func.return %arg0 : tensor<4x8xf32>
}


// Test output_direct_from_input_with_src_set

// It suffices to check with src_set, since if there's a use_set on func args,
// there will be a src_set (this is covered by tests for that).
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<4x8xf32> {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m2", "m3">}) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %arg0 {{.*}}mesh_tensor<"m2"
  // CHECK-NEXT: return %[[ASSIGN]]

  func.return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @operand_returned_multiple_times()
func.func @operand_returned_multiple_times() -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[CONST:.*]] = stablehlo.constant
  // CHECK-NEXT: %[[ASSIGN_1:.*]] = mpmd.assign {origin = "inferred_out"} %[[CONST]] {{.*}}mesh_tensor<"m2"

  // CHECK-NEXT: %[[ASSIGN_2:.*]] = mpmd.assign {origin = "inferred_out"} %[[CONST]] {{.*}}mesh_tensor<"m2"
  // CHECK-NEXT: return %[[ASSIGN_1]], %[[ASSIGN_2]]

  %0 = stablehlo.constant {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">} dense<1.0> : tensor<4x8xf32>
  func.return %0, %0 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @operand_returned_multiple_times_and_assigned(%arg0: tensor<4x8xf32>)
func.func @operand_returned_multiple_times_and_assigned(%arg0: tensor<4x8xf32>)
   -> (tensor<4x8xf32>, !mesh_2_tensor_4_8_f32, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[CONST:.*]] = stablehlo.constant
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign %[[CONST]]
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[CONST]]

  // CHECK-NEXT: %[[ASSIGN_1:.*]] = mpmd.assign {origin = "inferred_out"} %[[CONST]] {{.*}}mesh_tensor<"m2"

  // CHECK-NEXT: %[[ASSIGN_2:.*]] = mpmd.assign {origin = "inferred_out"} %[[ADD]] {{.*}}mesh_tensor<"m3"
  // CHECK-NEXT: return %[[ASSIGN_1]], %[[ASSIGN]], %[[ASSIGN_2]]

  %0 = stablehlo.constant {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">} dense<1.0> : tensor<4x8xf32>
  %1 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  %2 = stablehlo.add %arg0, %0 {mpmd.src_set = #mpmd.meshes_with_origins<"m3">} : tensor<4x8xf32>
  func.return %0, %1, %2 : tensor<4x8xf32>, !mesh_2_tensor_4_8_f32, tensor<4x8xf32>
}

// CHECK-LABEL: func @unused_computation_without_src_set(%arg0: tensor<4x8xf32>)
func.func @unused_computation_without_src_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT: %[[ASSIGN_ADD:.*]] = mpmd.assign {origin = "inferred_unused"} %[[ADD]] {{.*}}mesh_tensor<"m1"
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %arg0 {{.*}}mesh_tensor<"m1"
  // CHECK-NEXT: return %[[ASSIGN]]

  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  // Note we return arg0 and so %0 is unused
  func.return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @unused_computation_with_src_set(%arg0: tensor<4x8xf32>)
func.func @unused_computation_with_src_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  // CHECK:      %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT: %[[ASSIGN_ADD:.*]] = mpmd.assign {origin = "inferred_unused"} %[[ADD]] {{.*}}mesh_tensor<"m2"
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %arg0 {{.*}}mesh_tensor<"m1"
  // CHECK-NEXT: return %[[ASSIGN]]

  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">} : tensor<4x8xf32>
  // Note we return arg0 and so %0 is unused
  func.return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @op_with_no_result_without_src_set(%arg0: tensor<4x8xf32>)
func.func @op_with_no_result_without_src_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
// CHECK-NEXT:  %[[ASSIGN:.*]] = mpmd.assign %arg0
// CHECK-NEXT:  mpmd.fragment<mesh="m1", origin=[]> (%[[ASSIGN]]) (%arg1
// CHECK-NEXT:    sdy.sharding_group %arg1
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }

  // sdy.sharding_group has no results. So it gets wrapped in a fragment.
  sdy.sharding_group %arg0 group_id=0 : tensor<4x8xf32>
  func.return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @op_with_no_result_with_src_set(%arg0: tensor<4x8xf32>)
func.func @op_with_no_result_with_src_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
// CHECK-NEXT:  %[[ASSIGN:.*]] = mpmd.assign %arg0
// CHECK-NEXT:  mpmd.fragment<mesh="m2", origin=[]> (%[[ASSIGN]]) (%arg1
// CHECK-NEXT:    sdy.sharding_group %arg1
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }

  // sdy.sharding_group has no results. So it gets wrapped in a fragment.
  sdy.sharding_group %arg0 group_id=0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>
  func.return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @op_with_no_result_with_multi_src_set(%arg0: tensor<4x8xf32>)
func.func @op_with_no_result_with_multi_src_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
// CHECK-NEXT:  %[[ASSIGN_2:.*]] = mpmd.assign %arg0
// CHECK-NEXT:  mpmd.fragment<mesh="m2", origin=[]> (%[[ASSIGN_2]]) (%arg1
// CHECK-NEXT:    sdy.sharding_group %arg1
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[ASSIGN_3:.*]] = mpmd.assign %arg0
// CHECK-NEXT:  mpmd.fragment<mesh="m3", origin=[]> (%[[ASSIGN_3]]) (%arg1
// CHECK-NEXT:    sdy.sharding_group %arg1
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[ASSIGN:.*]] = mpmd.assign {origin = "inferred_out"} %arg0
// CHECK-NEXT:  return %[[ASSIGN]]

  // sdy.sharding_group has no results. So it gets wrapped in a fragment.
  sdy.sharding_group %arg0 group_id=0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">} : tensor<4x8xf32>
  func.return %arg0 : tensor<4x8xf32>
}


// CHECK-LABEL: func @assign_broadcast_operand
func.func @assign_broadcast_operand(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>  attributes {
  "topology"=#mpmd.topology<
    <"m0": <["x"=2]>>, <"m1": <["x"=2]>>
  >}
{
  // CHECK-NEXT: %[[PROD:.*]] = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = {{.*}}"m1", "m0">}
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "broadcast"} %[[PROD]] {{.*}}m1{{.*}}
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign %[[ASSIGN]]
  // CHECK-NEXT: %[[BCAST:.*]] = mpmd.broadcast %[[UNASSIGN]]
  // CHECK-NEXT: assign {origin = "inferred_out"} %[[BCAST]]
  %prod = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m0">} : tensor<4x8xf32>
  // Although the op is not actually used in m1, we set the use_set attribute.
  %0 = mpmd.broadcast {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %prod : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @assign_broadcast_operand
func.func @assign_broadcast_operand_empty_intersection(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>  attributes {
  "topology"=#mpmd.topology<
    <"m0": <["x"=2]>>, <"m1": <["x"=2]>>
  >}
{
  // CHECK-NEXT: %[[PROD:.*]] = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = {{.*}}"m0">}
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "broadcast"} %[[PROD]] {{.*}}m0{{.*}}
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign %[[ASSIGN]]
  // CHECK-NEXT: %[[BCAST:.*]] = mpmd.broadcast %[[UNASSIGN]]
  // CHECK-NEXT: assign {origin = "inferred_out"} %[[BCAST]] {{.*}}m1{{.*}}
  %prod = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m0">} : tensor<4x8xf32>
  // Although the op is not actually used in m1, we set the use_set attribute.
  %0 = mpmd.broadcast {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %prod : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @assign_broadcast_empty_use_set
func.func @assign_broadcast_empty_use_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>  attributes {
  "topology"=#mpmd.topology<
    <"m0": <["x"=2]>>, <"m1": <["x"=2]>>
  >}
{
  // CHECK-NEXT: %[[PROD:.*]] = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = {{.*}}"m0">}
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "broadcast"} %[[PROD]] {{.*}}m0{{.*}}
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign %[[ASSIGN]]
  // CHECK-NEXT: %[[BCAST:.*]] = mpmd.broadcast %[[UNASSIGN]]
  %prod = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m0">} : tensor<4x8xf32>
  %0 = mpmd.broadcast %prod : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @assign_reduce_operand
func.func @assign_reduce_operand(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>  attributes {
  "topology"=#mpmd.topology<
    <"m0": <["x"=2]>>, <"m1": <["x"=2]>>
  >}
{
  // CHECK-NEXT: %[[PROD:.*]] = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = {{.*}}"m1", "m0">}
  // CHECK-NEXT: %[[PROD1:.*]] = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = {{.*}}"m1", "m0">}
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "reduce"} %[[PROD]] {{.*}}m1{{.*}}
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign %[[ASSIGN]]
  // CHECK-NEXT: %[[ASSIGN1:.*]] = mpmd.assign {origin = "reduce"} %[[PROD1]] {{.*}}m1{{.*}}
  // CHECK-NEXT: %[[UNASSIGN1:.*]] = mpmd.unassign %[[ASSIGN1]]
  // CHECK-NEXT: %[[REDUCE:.*]] = mpmd.reduce<add> %[[UNASSIGN]], %[[UNASSIGN1]]
  // CHECK-NEXT: assign {origin = "inferred_out"} %[[REDUCE]]
  %prod = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m0">} : tensor<4x8xf32>
  %prod1 = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m0">} : tensor<4x8xf32>
  %0 = mpmd.reduce<add> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %prod, %prod1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @assign_reduce_operand_empty_intersection
func.func @assign_reduce_operand_empty_intersection(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>  attributes {
  "topology"=#mpmd.topology<
    <"m0": <["x"=2]>>, <"m1": <["x"=2]>>
  >}
{
  // CHECK-NEXT: %[[PROD:.*]] = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = {{.*}}"m0">}
  // CHECK-NEXT: %[[PROD1:.*]] = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = {{.*}}"m1">}
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "reduce"} %[[PROD]] {{.*}}m0{{.*}}
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign %[[ASSIGN]]
  // CHECK-NEXT: %[[ASSIGN1:.*]] = mpmd.assign {origin = "reduce"} %[[PROD1]] {{.*}}m1{{.*}}
  // CHECK-NEXT: %[[UNASSIGN1:.*]] = mpmd.unassign %[[ASSIGN1]]
  // CHECK-NEXT: %[[REDUCE:.*]] = mpmd.reduce<add> %[[UNASSIGN]], %[[UNASSIGN1]]
  // CHECK-NEXT: assign {origin = "inferred_out"} %[[REDUCE]] {{.*}}m1{{.*}}
  %prod = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m0">} : tensor<4x8xf32>
  %prod1 = stablehlo.multiply %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
  %0 = mpmd.reduce<add> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %prod, %prod1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @call_op_unused_op(
func.func @call_op_unused_op(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> (tensor<3x5xf32>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["x"=2]>>>}
{
  %0 = mpmd.call @call_op_unused_op_f(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// CHECK: func.func private @call_op_unused_op_f(
func.func private @call_op_unused_op_f(%arg0: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["x"=2]>>>}
{
  // The unused add op gets use_set = #mpmd.meshes_with_origins<"m1"> because "m1" is the default mesh
  // (since it is the first mesh in the topology).
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: %[[ASSIGN_ADD:.*]] = mpmd.assign {origin = "inferred_unused"} %[[ADD]] {{.*}}mesh_tensor<"m1"
  %0 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  return %arg0 : tensor<3x5xf32>
}

// CHECK-LABEL: func.func @call_op_unused_result(
func.func @call_op_unused_result(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> (tensor<3x5xf32>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["x"=2]>>>}
{
  %0:2 = mpmd.call @call_op_unused_result_f(%arg0, %arg1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  return %0#0 : tensor<3x5xf32>
}

// CHECK:     func.func private @call_op_unused_result_f(
// CHECK-SAME: %arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>
// CHECK-SAME: -> (tensor<3x5xf32>, tensor<3x5xf32> {mpmd.use_set = {{.*}}"m1"["inferred_unused_callee_out"]>})
func.func private @call_op_unused_result_f(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> (tensor<3x5xf32>, tensor<3x5xf32>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["x"=2]>>>}
{
  return %arg0, %arg1 : tensor<3x5xf32>, tensor<3x5xf32>
}

// CHECK-LABEL: func.func @call_op_unused_input(
func.func @call_op_unused_input(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> (tensor<3x5xf32>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["x"=2]>>>}
{
  %0 = mpmd.call @call_op_unused_input_f(%arg0, %arg1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// CHECK: func.func private @call_op_unused_input_f(
// CHECK-SAME: %arg0: tensor<3x5xf32>,
// CHECK-SAME: %arg1: tensor<3x5xf32> {mpmd.src_set = {{.*}}"m3", "m2">, mpmd.use_set = {{.*}}"m2"["inferred_unused_callee_in"]>}
func.func private @call_op_unused_input_f(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32> {mpmd.src_set = #mpmd.meshes_with_origins<"m3", "m2">})
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["x"=2]>>>}
{
  return %arg0 : tensor<3x5xf32>
}

// CHECK: func.func @broadcast_with_empty_src_set(
func.func @broadcast_with_empty_src_set(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.constant {mpmd.src_set = #mpmd.meshes_with_origins<>} dense<1.0> : tensor<4x8xf32>
  // CHECK: %[[ASSIGN:.*]] = mpmd.assign {{.*}}m2{{.*}}
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign %[[ASSIGN]]
  // CHECK-NEXT: %[[BCAST:.*]] = mpmd.broadcast %[[UNASSIGN]]
  %b = mpmd.broadcast {mpmd.use_set = #mpmd.meshes_with_origins<"m3", "m2">} %0 : tensor<4x8xf32>
  func.return %b : tensor<4x8xf32>
}

// CHECK: func.func @broadcast_with_empty_src_set_and_empty_use_set(
func.func @broadcast_with_empty_src_set_and_empty_use_set(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.constant {mpmd.src_set = #mpmd.meshes_with_origins<>} dense<1.0> : tensor<4x8xf32>
  // CHECK: %[[ASSIGN:.*]] = mpmd.assign {{.*}}m1{{.*}}
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign %[[ASSIGN]]
  // CHECK-NEXT: %[[BCAST:.*]] = mpmd.broadcast %[[UNASSIGN]]
  %b = mpmd.broadcast %0 : tensor<4x8xf32>
  func.return %b : tensor<4x8xf32>
}

// CHECK-LABEL: func @region_op_region_is_skipped
// Region op unused ops should not be given a use_set.
func.func @region_op_region_is_skipped(%arg0: tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %0 = stablehlo.constant dense<1> : tensor<i32>
  // CHECK: stablehlo.while
  // CHECK-NOT: mpmd.assign
  // CHECK: stablehlo.return
  // CHECK-NOT: mpmd.assign
  // CHECK: stablehlo.return
  // CHECK: mpmd.assign
  %5:2 = stablehlo.while(%iterArg_0 = %0, %iterArg_1 = %arg0) : tensor<i32>, tensor<4x8xf32>
   cond {
    %6 = "stablehlo.compare"(%iterArg_0, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%6) : (tensor<i1>) -> ()
  } do {
    %8 = stablehlo.add %iterArg_1, %iterArg_1 : tensor<4x8xf32>
    %9 = stablehlo.add %iterArg_1, %iterArg_1 : tensor<4x8xf32> // unused op should not be given a use_set
    "stablehlo.return"(%iterArg_0, %8) : (tensor<i32>, tensor<4x8xf32>) -> ()
  }
  %6 = mpmd.assign %5#1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32

  func.return %6 : !mesh_1_tensor_4_8_f32
}
