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
// CHECK-NEXT:    %[[IOTA:.*]] = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : tensor<8x8xi32>
// CHECK-NEXT:    %[[LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
// CHECK-NEXT:    %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<i32>) -> tensor<8x8xi32>
// CHECK-NEXT:    %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
// CHECK-NEXT:    %[[BCAST_CST:.*]] = stablehlo.broadcast_in_dim %[[CST]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_CST]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : tensor<8x8xi1>, tensor<8x8xf32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[SELECT]], %[[CST]], low = [1, 0], high = [0, 0], interior = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<16x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4, [{}, {}]> : tensor<16x8xf32>
// CHECK-NEXT:    %[[TRIM:.*]] = stablehlo.slice %[[ALL_GATHER]] [0:15, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {}]>]>} : (tensor<16x8xf32>) -> tensor<15x8xf32>
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

// The pad below is padded for divisibility along dim 0 (9 -> 12) with its own
// zero padding value, so the trailing rows 9-11 are already zero. The pad
// pattern registers this zero padding in the padding cache, so the dot_general
// pattern can contract over dim 0 without inserting an iota/compare/select mask
// on the lhs. Without caching, the lhs would be masked with a stablehlo.select.
// CHECK-LABEL: func @pad_then_dot_general
// CHECK-SAME:    %[[ARG0:.*]]: tensor<7x7xf32>, %[[ARG1:.*]]: tensor<9x5xf32>
// CHECK-DAG:     %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [1, 0], high = [4, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<7x7xf32>, tensor<f32>) -> tensor<12x7xf32>
// CHECK-DAG:     %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD1:.*]] = stablehlo.pad %[[ARG1]], %[[CST1]], low = [0, 0], high = [3, 0], interior = [0, 0] : (tensor<9x5xf32>, tensor<f32>) -> tensor<12x5xf32>
// CHECK-DAG:     %[[SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD1]] out_sharding=<@mesh_4, [{"x"}, {}]> : tensor<12x5xf32>
// CHECK-NOT:     stablehlo.select
// The dot_general consumes %[[PAD0]] directly, i.e. no mask is applied.
// CHECK:         %[[DOT:.*]] = stablehlo.dot_general %[[PAD0]], %[[SLICE]], contracting_dims = [0] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {}]>]>} : (tensor<12x7xf32>, tensor<12x5xf32>) -> tensor<7x5xf32>
// CHECK-NEXT:    return %[[DOT]] : tensor<7x5xf32>
func.func @pad_then_dot_general(%arg0: tensor<7x7xf32>, %arg1: tensor<9x5xf32>) -> tensor<7x5xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [1, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<7x7xf32>, tensor<f32>) -> tensor<9x7xf32>
  %arg1_sharded = sdy.all_slice [{"x"}, {}] %arg1 out_sharding=<@mesh_4, [{"x"}, {}]> : tensor<9x5xf32>
  %1 = stablehlo.dot_general %0, %arg1_sharded, contracting_dims = [0] x [0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{}, {}]>]>} : (tensor<9x7xf32>, tensor<9x5xf32>) -> tensor<7x5xf32>
  return %1 : tensor<7x5xf32>
}

// Same idea for stablehlo.reduce: the pad result is sharded along the
// reduction dim and padded for divisibility (7 -> 8) with zero. Zero is the
// identity of add, so the reduce consumes the padded tensor directly instead
// of masking column 7 with a stablehlo.select.
// CHECK-LABEL: func @zero_pad_then_reduce_add
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x5xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [0, 3], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : (tensor<4x5xf32>, tensor<f32>) -> tensor<4x8xf32>
// CHECK-NEXT:    %[[REDUCE:.*]] = stablehlo.reduce(%[[PAD]] init: %[[CST]]) applies stablehlo.add across dimensions = [1]
// CHECK-SAME:    : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[REDUCE]] : tensor<4xf32>
func.func @zero_pad_then_reduce_add(%arg0: tensor<4x5xf32>) -> tensor<4xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [0, 0], high = [0, 2], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{}, {"x"}]>]>} : (tensor<4x5xf32>, tensor<f32>) -> tensor<4x7xf32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{}]>]>} : (tensor<4x7xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// Negative case: the pad value is 1, so the padded column 7 holds 1, which is
// not the identity of add. The mask must stay: column 7 (iota >= 7) is replaced
// with the init value 0 before the reduce. The real padding in columns 5-6
// (iota < 7) is kept, since those 1s are part of the original tensor.
// CHECK-LABEL: func @one_pad_then_reduce_add
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x5xf32>
// CHECK-DAG:     %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[ONE]], low = [0, 0], high = [0, 3], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : (tensor<4x5xf32>, tensor<f32>) -> tensor<4x8xf32>
// CHECK:         %[[IOTA:.*]] = stablehlo.iota dim = 1
// CHECK:         %[[LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
// CHECK:         %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %{{.*}}
// CHECK:         %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[PAD]], %{{.*}}
// CHECK:         %[[REDUCE:.*]] = stablehlo.reduce(%[[SELECT]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [1]
// CHECK-SAME:    : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[REDUCE]] : tensor<4xf32>
func.func @one_pad_then_reduce_add(%arg0: tensor<4x5xf32>) -> tensor<4xf32> {
  %one = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.pad %arg0, %one, low = [0, 0], high = [0, 2], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{}, {"x"}]>]>} : (tensor<4x5xf32>, tensor<f32>) -> tensor<4x7xf32>
  %1 = stablehlo.reduce(%0 init: %zero) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{}]>]>} : (tensor<4x7xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
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

// CHECK-LABEL: func @interior_padding_divisible_result(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<16x8xf32> {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[IOTA:.*]] = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : tensor<8x8xi32>
// CHECK-NEXT:    %[[LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
// CHECK-NEXT:    %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<i32>) -> tensor<8x8xi32>
// CHECK-NEXT:    %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
// CHECK-NEXT:    %[[BCAST_CST:.*]] = stablehlo.broadcast_in_dim %[[CST]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<f32>) -> tensor<8x8xf32>
// CHECK-NEXT:    %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_CST]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : tensor<8x8xi1>, tensor<8x8xf32>
// CHECK-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[SELECT]], %[[CST]], low = [1, 0], high = [0, 0], interior = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<16x8xf32>
// CHECK-NEXT:    %[[ALL_GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4, [{}, {}]> : tensor<16x8xf32>
// CHECK-NEXT:    return %[[ALL_GATHER]] : tensor<16x8xf32>
// CHECK-NEXT:  }
func.func @interior_padding_divisible_result(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) -> tensor<16x8xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.pad %0, %cst, low = [1, 0], high = [2, 0], interior = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4, [{"x"}, {}]>]>} : (tensor<7x8xf32>, tensor<f32>) -> tensor<16x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4, [{}, {}]> : tensor<16x8xf32>
  return %2 : tensor<16x8xf32>
}
