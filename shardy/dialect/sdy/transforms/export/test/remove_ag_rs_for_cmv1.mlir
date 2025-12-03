// RUN: sdy_opt %s -sdy-remove-all-gather-reduce-scatter-for-cmv1 | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @single_all_gather
func.func @single_all_gather(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %0 = sdy.all_gather [{}, {"y"}] %arg0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %0
  %0 = sdy.all_gather [{}, {"y"}] %arg0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @all_gather_dot
func.func @all_gather_dot(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: return %0
  %0 = sdy.all_gather [{"x"}, {}] %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<16x32xf32>
  %1 = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

// CHECK-LABEL: func @all_gather_multiple_uses
func.func @all_gather_multiple_uses(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %0 = sdy.all_gather [{"x"}, {}] %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<16x32xf32>
  // CHECK: %1 = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: return %0, %1
  %0 = sdy.all_gather [{"x"}, {}] %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<16x32xf32>
  %1 = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0, %1 : tensor<16x32xf32>, tensor<8x32xf32>
}

// CHECK-LABEL: func @single_reduce_scatter
func.func @single_reduce_scatter(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %0 = sdy.reduce_scatter [{"x"}, {}] %arg0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %0
  %0 = sdy.reduce_scatter [{"x"}, {}] %arg0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_reduce_scatter
func.func @dot_reduce_scatter(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
    %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: return %0
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %1 = sdy.reduce_scatter [{"x"}, {}] %0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

// CHECK-LABEL: func @dot_reduce_scatter_with_unreduced_axes
func.func @dot_reduce_scatter_with_unreduced_axes(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>},
    %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>})
    -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) {
  // CHECK: %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x", "y"}>]>}
  // CHECK-NEXT: %1 = sdy.reduce_scatter [{"x"}, {}] %0 out_sharding=<@mesh, [{"x"}, {}], unreduced={"y"}> : tensor<8x32xf32>
  // CHECK-NEXT: return %1
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x", "y"}>]>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %1 = sdy.reduce_scatter [{"x"}, {}] %0 out_sharding=<@mesh, [{"x"}, {}], unreduced={"y"}> : tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

// CHECK-LABEL: func @all_gather_dot_reduce_scatter
func.func @all_gather_dot_reduce_scatter(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
    -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK: %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %1 = sdy.reduce_scatter [{"x"}, {}] %0 out_sharding=<@mesh, [{"x"}, {"y"}]> : tensor<8x32xf32>
  // CHECK-NEXT: return %0
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  %1 = stablehlo.dot %0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %2 = sdy.reduce_scatter [{"x"}, {}] %1 out_sharding=<@mesh, [{"x"}, {"y"}]> : tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}
