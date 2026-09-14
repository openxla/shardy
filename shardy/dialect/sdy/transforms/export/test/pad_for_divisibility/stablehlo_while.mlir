// RUN: sdy_opt %s -sdy-pad-for-divisibility -split-input-file | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func.func @while_indivisible(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<7x8xf32>) -> tensor<7x8xf32> {
func.func @while_indivisible(%arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<7x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>

  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>

  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>

  // CHECK-NEXT: %[[C10:.*]] = stablehlo.constant dense<10> : tensor<i32>
  %c10 = stablehlo.constant dense<10> : tensor<i32>

  // CHECK-NEXT: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %[[SLICE]], %iterArg_2 = %[[C0]]) : tensor<8x8xf32>, tensor<i32>
  // CHECK-SAME:     attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>, <@mesh_4_2, []>]>}
  // CHECK-NEXT: cond {
  // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare LT, %iterArg_2, %[[C10]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT:   stablehlo.return %[[COND]] : tensor<i1>
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[NEXT_I:.*]] = stablehlo.add %iterArg_2, %[[C1]] : tensor<i32>
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]], %[[NEXT_I]] : tensor<8x8xf32>, tensor<i32>
  // CHECK-NEXT: }
  %1:2 = stablehlo.while(%iterArg = %0, %iterArg_1 = %c0) : tensor<7x8xf32>, tensor<i32>
    attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>, <@mesh_4_2, []>]>}
    cond {
      %cond = stablehlo.compare LT, %iterArg_1, %c10 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %step = stablehlo.add %iterArg_1, %c1 : tensor<i32>
      %add = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<7x8xf32>
      stablehlo.return %add, %step : tensor<7x8xf32>, tensor<i32>
    }

  // CHECK-NEXT: %[[GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[WHILE]]#0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[TRIM:.*]] = stablehlo.slice %[[GATHER]] [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1#0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>

  // CHECK-NEXT: return %[[TRIM]] : tensor<7x8xf32>
  return %2 : tensor<7x8xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func.func @while_mixed_divisibility(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<7x8xf32>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<16x8xf32>) -> (tensor<7x8xf32>, tensor<16x8xf32>) {
func.func @while_mixed_divisibility(%arg0: tensor<7x8xf32>, %arg1: tensor<16x8xf32>) -> (tensor<7x8xf32>, tensor<16x8xf32>) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<7x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[SLICE0:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>

  // CHECK-NEXT: %[[SLICE1:.*]] = sdy.all_slice [{"x"}, {}] %[[ARG1]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_slice [{"x"}, {}] %arg1 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<16x8xf32>

  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>

  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>

  // CHECK-NEXT: %[[C10:.*]] = stablehlo.constant dense<10> : tensor<i32>
  %c10 = stablehlo.constant dense<10> : tensor<i32>

  // CHECK-NEXT: %[[WHILE:.*]]:3 = stablehlo.while(%iterArg = %[[SLICE0]], %iterArg_2 = %[[SLICE1]], %iterArg_3 = %[[C0]]) : tensor<8x8xf32>, tensor<16x8xf32>, tensor<i32>
  // CHECK-SAME:     attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>, <@mesh_4_2, [{"x"}, {}]>, <@mesh_4_2, []>]>}
  // CHECK-NEXT: cond {
  // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare LT, %iterArg_3, %[[C10]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT:   stablehlo.return %[[COND]] : tensor<i1>
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[NEXT_I:.*]] = stablehlo.add %iterArg_3, %[[C1]] : tensor<i32>
  // CHECK-NEXT:   %[[ADD0:.*]] = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT:   %[[ADD1:.*]] = stablehlo.add %iterArg_2, %iterArg_2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<16x8xf32>
  // CHECK-NEXT:   stablehlo.return %[[ADD0]], %[[ADD1]], %[[NEXT_I]] : tensor<8x8xf32>, tensor<16x8xf32>, tensor<i32>
  // CHECK-NEXT: }
  %2:3 = stablehlo.while(%iterArg0 = %0, %iterArg1 = %1, %iterArg_2 = %c0) : tensor<7x8xf32>, tensor<16x8xf32>, tensor<i32>
    attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>, <@mesh_4_2, [{"x"}, {}]>, <@mesh_4_2, []>]>}
    cond {
      %cond = stablehlo.compare LT, %iterArg_2, %c10 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %step = stablehlo.add %iterArg_2, %c1 : tensor<i32>
      %add0 = stablehlo.add %iterArg0, %iterArg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<7x8xf32>
      %add1 = stablehlo.add %iterArg1, %iterArg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<16x8xf32>
      stablehlo.return %add0, %add1, %step : tensor<7x8xf32>, tensor<16x8xf32>, tensor<i32>
    }

  // CHECK-NEXT: %[[GATHER0:.*]] = sdy.all_gather [{"x"}, {}] %[[WHILE]]#0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[TRIM:.*]] = stablehlo.slice %[[GATHER0]] [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %3 = sdy.all_gather [{"x"}, {}] %2#0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>

  // CHECK-NEXT: %[[GATHER1:.*]] = sdy.all_gather [{"x"}, {}] %[[WHILE]]#1 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<16x8xf32>
  %4 = sdy.all_gather [{"x"}, {}] %2#1 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<16x8xf32>

  // CHECK-NEXT: return %[[TRIM]], %[[GATHER1]] : tensor<7x8xf32>, tensor<16x8xf32>
  return %3, %4 : tensor<7x8xf32>, tensor<16x8xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func.func private @subroutine_while(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
func.func private @subroutine_while(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>

  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>

  // CHECK-NEXT: %[[C10:.*]] = stablehlo.constant dense<10> : tensor<i32>
  %c10 = stablehlo.constant dense<10> : tensor<i32>

  // CHECK-NEXT: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %[[ARG0]], %iterArg_2 = %[[C0]]) : tensor<8x8xf32>, tensor<i32>
  // CHECK-SAME:     attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>, <@mesh_4_2, []>]>}
  // CHECK-NEXT: cond {
  // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare LT, %iterArg_2, %[[C10]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT:   stablehlo.return %[[COND]] : tensor<i1>
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[NEXT_I:.*]] = stablehlo.add %iterArg_2, %[[C1]] : tensor<i32>
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]], %[[NEXT_I]] : tensor<8x8xf32>, tensor<i32>
  // CHECK-NEXT: }
  %1:2 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %c0) : tensor<7x8xf32>, tensor<i32>
    attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>, <@mesh_4_2, []>]>}
    cond {
      %cond = stablehlo.compare LT, %iterArg_1, %c10 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %step = stablehlo.add %iterArg_1, %c1 : tensor<i32>
      %add = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<7x8xf32>
      stablehlo.return %add, %step : tensor<7x8xf32>, tensor<i32>
    }

  // CHECK-NEXT: return %[[WHILE]]#0 : tensor<8x8xf32>
  return %1#0 : tensor<7x8xf32>
}
