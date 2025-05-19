// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @broadcast_in_dim
func.func @broadcast_in_dim(%arg0: tensor<2x3x5x1x7xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}, {}]>}) -> tensor<2x5x3x11x7x13xf32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}, {}, {}]> : tensor<2x3x5x1x7xf32>
  // CHECK-NEXT: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %[[RESHARD]], dims = [0, 2, 1, 3, 4] : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  // CHECK-NEXT: return %[[BROADCAST_IN_DIM]] : tensor<2x5x3x11x7x13xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 1, 3, 4] : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  return %0 :  tensor<2x5x3x11x7x13xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_input_output_different
func.func @broadcast_in_dim_input_output_different(%arg0: tensor<2x3x5x1x7xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}, {}]>}) -> (tensor<2x5x3x11x7x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}, {}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {"x"}, {}, {}]> : tensor<2x3x5x1x7xf32>
  // CHECK-NEXT: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %[[RESHARD]], dims = [0, 2, 1, 3, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}, {}, {}, {"y"}]>]>} : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  // CHECK-NEXT: return %[[BROADCAST_IN_DIM]] : tensor<2x5x3x11x7x13xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 1, 3, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}, {}, {}, {"y"}]>]>} : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  return %0 :  tensor<2x5x3x11x7x13xf32>
}
