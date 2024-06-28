// RUN: sdy_opt %s 2>&1 | FileCheck %s

// CHECK: sdy.mesh @foo = <"a"=2, "b"=4>
sdy.mesh @foo = <"a"=2, "b"=4>

// CHECK: sdy.mesh @bar = <"a"=4, "b"=2>
sdy.mesh @bar = <"a"=4, "b"=2>

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
