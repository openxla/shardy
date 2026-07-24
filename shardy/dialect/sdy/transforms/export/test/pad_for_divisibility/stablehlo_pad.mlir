// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4 = <["x"=4]>

// CHECK-LABEL: func @extend_padding(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<10x8xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [1, 0], high = [3, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<12x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4, [{}, {}]> : tensor<12x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:10, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {}]>]>} : (tensor<12x8xf32>) -> tensor<10x8xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<10x8xf32>
// CHECK-NEXT:  }
func.func @extend_padding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<10x8xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.pad %0, %cst, low = [1, 0], high = [2, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<7x8xf32>, tensor<f32>) -> tensor<10x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4, [{}, {}]> : tensor<10x8xf32>
  return %2 : tensor<10x8xf32>
}

// CHECK-LABEL: func @interior_padding(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<15x8xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [1, 0], high = [4, 0], interior = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<20x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4, [{}, {}]> : tensor<20x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:15, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {}]>]>} : (tensor<20x8xf32>) -> tensor<15x8xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<15x8xf32>
// CHECK-NEXT:  }
func.func @interior_padding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<15x8xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.pad %0, %cst, low = [1, 0], high = [1, 0], interior = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<7x8xf32>, tensor<f32>) -> tensor<15x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4, [{}, {}]> : tensor<15x8xf32>
  return %2 : tensor<15x8xf32>
}

// CHECK-LABEL: func @no_op(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<12x8xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [2, 0], high = [2, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<12x8xf32>
// CHECK-NEXT:    return %[[PAD]] : tensor<12x8xf32>
// CHECK-NEXT:  }
func.func @no_op(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<12x8xf32> {
  %0 = stablehlo.slice %arg0 [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.pad %0, %cst, low = [2, 0], high = [2, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<12x8xf32>
  return %1 : tensor<12x8xf32>
}

// CHECK-LABEL: func @pad_then_dot_general
// CHECK-SAME:    %[[ARG0:.*]]: tensor<7x7xf32>, %[[ARG1:.*]]: tensor<9x5xf32>
// CHECK:         stablehlo.select
// CHECK:         stablehlo.dot_general
func.func @pad_then_dot_general(%arg0: tensor<7x7xf32>, %arg1: tensor<9x5xf32>) -> tensor<7x5xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [1, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<7x7xf32>, tensor<f32>) -> tensor<9x7xf32>
  %arg1_sharded = sdy.all_slice [{"x"}, {}] %arg1 out_sharding=<@mesh_4, [{"x"}, {}]> : tensor<9x5xf32>
  %1 = stablehlo.dot_general %0, %arg1_sharded, contracting_dims = [0] x [0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{}, {}]>]>} : (tensor<9x7xf32>, tensor<9x5xf32>) -> tensor<7x5xf32>
  return %1 : tensor<7x5xf32>
}

// CHECK-LABEL: func @negative_high_padding(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<7x8xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [1, 0], high = [-1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4, [{}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<7x8xf32>
// CHECK-NEXT:  }
func.func @negative_high_padding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<7x8xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.pad %0, %cst, low = [1, 0], high = [-1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<7x8xf32>, tensor<f32>) -> tensor<7x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4, [{}, {}]> : tensor<7x8xf32>
  return %2 : tensor<7x8xf32>
}

// CHECK-LABEL: func @negative_low_padding(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<7x8xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [-1, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4, [{}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<7x8xf32>
// CHECK-NEXT:  }
func.func @negative_low_padding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<7x8xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.pad %0, %cst, low = [-1, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<7x8xf32>, tensor<f32>) -> tensor<7x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4, [{}, {}]> : tensor<7x8xf32>
  return %2 : tensor<7x8xf32>
}

// CHECK-LABEL: func @negative_high_padding_become_noop(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<6x8xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [0, 0], high = [0, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4, [{}, {}]> : tensor<8x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:6, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<6x8xf32>
// CHECK-NEXT:    return %[[TRIM]] : tensor<6x8xf32>
// CHECK-NEXT:  }
func.func @negative_high_padding_become_noop(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<6x8xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.pad %0, %cst, low = [0, 0], high = [-1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<7x8xf32>, tensor<f32>) -> tensor<6x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4, [{}, {}]> : tensor<6x8xf32>
  return %2 : tensor<6x8xf32>
}
