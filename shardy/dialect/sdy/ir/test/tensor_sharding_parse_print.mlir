// RUN: sdy_opt %s 2>&1 | FileCheck %s

// CHECK: sdy.mesh @foo = <["a"=2, "b"=4]>
sdy.mesh @foo = <["a"=2, "b"=4]>

// CHECK: sdy.mesh @bar = <["a"=4, "b"=2]>
sdy.mesh @bar = <["a"=4, "b"=2]>

// CHECK: sdy.mesh @maximal_mesh = <[], device_ids=[0]>
sdy.mesh @maximal_mesh = <[], device_ids=[0]>

// CHECK-LABEL: func @no_results
func.func @no_results(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: return {sdy.sharding = #sdy.sharding_per_value<[]>} %arg0
  return {sdy.sharding = #sdy.sharding_per_value<[]>} %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @rank_0
func.func @rank_0(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [], replicated={"b"}>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [], replicated={"b"}>]>} : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @no_replicated_axes
func.func @no_replicated_axes(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @replicated_axes
func.func @replicated_axes(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a", "b"}>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a", "b"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @unreduced_axes
func.func @unreduced_axes(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced={"b", "a"}>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced={"b", "a"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @replicated_and_unreduced_axes
func.func @replicated_and_unreduced_axes(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, unreduced={"b"}>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, unreduced={"b"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @unreduced_and_replicated_axes
func.func @unreduced_and_replicated_axes(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, unreduced={"b"}>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced={"b"}, replicated={"a"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @inlined_mesh
func.func @inlined_mesh(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @inlined_mesh_and_ref
func.func @inlined_mesh_and_ref(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.optimization_barrier
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<mesh<["a"=2, "b"=4]>, [{"a"}, {}]>, <@foo, [{"a"}, {}]>]>
  %0:2 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=2, "b"=4]>, [{"a"}, {}]>, <@foo, [{"a"}, {}]>]>} %arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>
  return %0#0, %0#1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @inlined_mesh_non_iota_device_ids
func.func @inlined_mesh_non_iota_device_ids(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<mesh<["x"=2], device_ids=[1, 0]>, [{"x"}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2], device_ids=[1, 0]>, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @inlined_mesh_iota_device_ids
func.func @inlined_mesh_iota_device_ids(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<mesh<["x"=2]>, [{"x"}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2], device_ids=[0, 1]>, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @tensor_with_open_dimension_shardings
func.func @tensor_with_open_dimension_shardings(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a", ?}, {?}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @tensor_with_closed_dimension_shardings
func.func @tensor_with_closed_dimension_shardings(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @tensor_with_open_closed_dimension_shardings
func.func @tensor_with_open_closed_dimension_shardings(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a", ?}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a", ?}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @open_dim_sharding_with_multiple_axes
func.func @open_dim_sharding_with_multiple_axes(%arg0 : tensor<16x8xf32>, %arg1 : tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a", "b", ?}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a", "b", ?}, {}]>]>} : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @tensor_with_sub_axes
func.func @tensor_with_sub_axes(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a"}, {"b":(2)2}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {"b":(2)2}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @replicated_sub_axis
func.func @replicated_sub_axis(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a"}, {}], replicated={"b":(2)2}>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}], replicated={"b":(2)2}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @unreduced_sub_axis
func.func @unreduced_sub_axis(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a"}, {}], unreduced={"b":(2)2}>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}], unreduced={"b":(2)2}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @dim_sharding_with_full_axis_between_sub_axes
func.func @dim_sharding_with_full_axis_between_sub_axes(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@bar, [{"a":(1)2, "b", "a":(2)2}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@bar, [{"a":(1)2, "b", "a":(2)2}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @dim_sharding_with_consecutive_sub_axes_in_reverse
func.func @dim_sharding_with_consecutive_sub_axes_in_reverse(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@bar, [{"a":(2)2, "a":(1)2}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@bar, [{"a":(2)2, "a":(1)2}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @sharding_with_priority
// CHECK-SAME{LITERAL}: #sdy.sharding<@foo, [{"a"}p0, {"b"}]>}
func.func @sharding_with_priority(%arg0 : tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@foo, [{"a"}p0, {"b"}]>}, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @op_sharding_with_priority
func.func @op_sharding_with_priority(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a"}, {"b"}p1]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {"b"}p1]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @op_sharding_with_multiple_priorities
func.func @op_sharding_with_multiple_priorities(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a"}p0, {"b"}p1]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}p0, {"b"}p1]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @dynamic_shaped_tensor_with_sharding
// CHECK-SAME:    %arg0: tensor<?x?xf32> {sdy.sharding = #sdy.sharding<@foo, [{}, {"a"}]>}
func.func @dynamic_shaped_tensor_with_sharding(%arg0: tensor<?x?xf32> {sdy.sharding = #sdy.sharding<@foo, [{}, {"a"}]>}) -> tensor<?x?xf32> {
  return %arg0 : tensor<?x?xf32>
}


// CHECK-LABEL: func @single_tuple
// CHECK-SAME:    %arg0: tensor<8x8xf32>) -> tuple<tensor<8x8xf32>> {
func.func @single_tuple(%arg0: tensor<8x8xf32>) -> tuple<tensor<8x8xf32>> {
  // CHECK-NEXT: stablehlo.custom_call @sdy_testonly(%arg0)
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>
  %0 = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {}]>]>} : (tensor<8x8xf32>) -> tuple<tensor<8x8xf32>>
  return %0 : tuple<tensor<8x8xf32>>
}

// CHECK-LABEL: func @maximal_sharding_no_results
// CHECK-SAME:      (%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @maximal_sharding_no_results(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, []>]>} : (tensor<8x8xf32>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<8x8xf32>
  stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, []>]>} : (tensor<8x8xf32>) -> ()
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @replicated_sharding_no_results
// CHECK-SAME:      (%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @replicated_sharding_no_results(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@foo, []>]>} : (tensor<8x8xf32>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<8x8xf32>
  stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@foo, []>]>} : (tensor<8x8xf32>) -> ()
  return %arg0 : tensor<8x8xf32>
}
