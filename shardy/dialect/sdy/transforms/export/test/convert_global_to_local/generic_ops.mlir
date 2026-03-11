// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2 = <["x"=2]>
// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func.func @func_returning_sharded_arg
// CHECK-SAME:    (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>})
func.func @func_returning_sharded_arg(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
  // CHECK-NEXT:  return %arg0 : tensor<8xf32>
  return %arg0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @func_with_dot_then_add
// CHECK-SAME:    (%arg0: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
// CHECK-SAME:    %arg1: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>},
// CHECK-SAME:    %arg2: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
// CHECK-SAME:    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
func.func @func_with_dot_then_add(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>},
  %arg2: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
  -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT:  %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[DOT]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : tensor<4x8xf32>
  %1 = stablehlo.add %0, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : tensor<8x32xf32>
  // CHECK-NEXT:  return %[[ADD]] : tensor<4x8xf32>
  return %1 : tensor<8x32xf32>
}

// CHECK-LABEL: func.func @sdy_all_reduce
// CHECK-SAME:    (%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>}) {
func.func @sdy_all_reduce(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>}) {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_2_4, [{}, {"y"}]> : tensor<16x8xf32>
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_2_4, [{}, {"y"}]> : tensor<16x32xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<16x8xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @stablehlo_reduce
// CHECK-SAME: (%[[ARG0:.*]]: tensor<32x8xi32>)
// CHECK-SAME: -> tensor<32xi32> {
func.func @stablehlo_reduce(%arg0: tensor<32x8xi32>)
    -> (tensor<32xi32>) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.reduce(%[[ARG0]] init: %[[CST]]) across dimensions = [1] : (tensor<32x8xi32>, tensor<i32>) -> tensor<32xi32>
  // CHECK-NEXT:  reducer(%[[ARG1:.*]]: tensor<i32>, %[[ARG2:.*]]: tensor<i32>) {
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %[[ARG1]], %[[ARG2]] : tensor<i32>
  // CHECK-NEXT:    %[[ADD_2:.*]] = stablehlo.add %[[ADD]], %[[ARG2]] : tensor<i32>
  // CHECK-NEXT:    stablehlo.return %[[ADD_2]] : tensor<i32>
  // CHECK-NEXT:  }
  %cst = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce"(%arg0, %cst) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      %2 = stablehlo.add %1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
  }) {
    dimensions = array<i64: 1>
  }: (tensor<32x8xi32>, tensor<i32>) -> tensor<32xi32>
  // CHECK-NEXT: return %[[RES]] : tensor<32xi32>
  return %0 : tensor<32xi32>
}
