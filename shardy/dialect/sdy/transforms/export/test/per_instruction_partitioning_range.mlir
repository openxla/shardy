// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning="filter='selectLow=0, selectHigh=0'" | FileCheck %s
// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning="filter='selectHigh=0, selectLow=0'" | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @selective_dot_range
// CHECK-SAME: (%[[LHS:.*]]: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %[[RHS:.*]]: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func @selective_dot_range(%lhs: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
                               %rhs: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[LHS]], %[[RHS]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {"y"}]>]
  // CHECK-SAME:   manual_axes={"x", "y"} (%arg2: tensor<4x32xf32>, %arg3: tensor<32x8xf32>) {
  // CHECK-NEXT:   %[[LOCAL_DOT:.*]] = stablehlo.dot %arg2, %arg3 : (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT:   sdy.return %[[LOCAL_DOT]] : tensor<4x8xf32>
  // CHECK-NEXT: } : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %dot = stablehlo.dot %lhs, %rhs {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>

  // CHECK: %[[ADD:.*]] = stablehlo.add %[[MANUAL]], %[[MANUAL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x16xf32>
  %add = stablehlo.add %dot, %dot {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x16xf32>

  // CHECK: return %[[ADD]] : tensor<8x16xf32>
  return %add : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @selective_add_range
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @selective_add_range(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK:      %[[ADD1:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG0]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg1: tensor<2x8xf32>, %arg2: tensor<2x8xf32>) {
  // CHECK-NEXT:   %[[LOCAL1:.*]] = stablehlo.add %arg1, %arg2 : tensor<2x8xf32>
  // CHECK-NEXT:   sdy.return %[[LOCAL1]] : tensor<2x8xf32>
  // CHECK-NEXT: } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %add1 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}: tensor<4x8xf32>

  // CHECK: %[[ADD2:.*]] = stablehlo.add %[[ADD1]], %[[ADD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xf32>
  %add2 = stablehlo.add %add1, %add1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}: tensor<4x8xf32>

  // CHECK: return %[[ADD2]] : tensor<4x8xf32>
  return %add2 : tensor<4x8xf32>
}
