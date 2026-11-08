// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh = <["x"=2]>

func.func @parallel_reduce_non_standard(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
  %arg1: tensor<4x4xi1> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
  %arg2: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
      tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>},
      tensor<4xi1> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
      tensor<4xi1> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>},
      tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
      tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>},
      tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
      tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  %c1_i32 = stablehlo.constant dense<1> : tensor<i32>
  %prod_rep = stablehlo.reduce(%arg0 init: %c1_i32) applies stablehlo.multiply across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}]>]>} : (tensor<4x4xi32>, tensor<i32>) -> tensor<4xi32>

  %prod_sharded = stablehlo.reduce(%arg0 init: %c1_i32) applies stablehlo.multiply across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}]>]>} : (tensor<4x4xi32>, tensor<i32>) -> tensor<4xi32>

  %true_i1 = stablehlo.constant dense<true> : tensor<i1>
  %all_rep = stablehlo.reduce(%arg1 init: %true_i1) applies stablehlo.and across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}]>]>} : (tensor<4x4xi1>, tensor<i1>) -> tensor<4xi1>

  %false_i1 = stablehlo.constant dense<false> : tensor<i1>
  %any_sharded = stablehlo.reduce(%arg1 init: %false_i1) applies stablehlo.or across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}]>]>} : (tensor<4x4xi1>, tensor<i1>) -> tensor<4xi1>

  %c0_i32 = stablehlo.constant dense<0> : tensor<i32>
  %xor_rep = stablehlo.reduce(%arg0 init: %c0_i32) applies stablehlo.xor across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}]>]>} : (tensor<4x4xi32>, tensor<i32>) -> tensor<4xi32>

  %inf_f32 = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %custom_sharded = stablehlo.reduce(%arg2 init: %inf_f32) across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}]>]>} : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %one = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %lt_l = stablehlo.compare LT, %lhs, %one : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %lt_r = stablehlo.compare LT, %rhs, %one : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %both_lt = stablehlo.and %lt_l, %lt_r : tensor<i1>
      %mx = stablehlo.maximum %lhs, %rhs : tensor<f32>
      %mn = stablehlo.minimum %lhs, %rhs : tensor<f32>
      %sel = stablehlo.select %both_lt, %mx, %mn : tensor<i1>, tensor<f32>
      stablehlo.return %sel : tensor<f32>
    }

  %clamped_sum = stablehlo.reduce(%arg0 init: %c0_i32) across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}]>]>} : (tensor<4x4xi32>, tensor<i32>) -> tensor<4xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %cap = stablehlo.constant dense<8> : tensor<i32>
      %sum = stablehlo.add %lhs, %rhs : tensor<i32>
      %clamped = stablehlo.minimum %sum, %cap : tensor<i32>
      stablehlo.return %clamped : tensor<i32>
    }

  %or_ending_in_add = stablehlo.reduce(%arg0 init: %c0_i32) across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}]>]>} : (tensor<4x4xi32>, tensor<i32>) -> tensor<4xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %zero = stablehlo.constant dense<0> : tensor<i32>
      %or_val = stablehlo.or %lhs, %rhs : tensor<i32>
      %res = stablehlo.add %or_val, %zero : tensor<i32>
      stablehlo.return %res : tensor<i32>
    }

  return %prod_rep, %prod_sharded, %all_rep, %any_sharded, %xor_rep, %custom_sharded, %clamped_sum, %or_ending_in_add
    : tensor<4xi32>, tensor<4xi32>, tensor<4xi1>, tensor<4xi1>, tensor<4xi32>, tensor<4xf32>, tensor<4xi32>, tensor<4xi32>
}

//--- part2.mlir

func.func @main() {
  %in_i32 = stablehlo.constant dense<[[2, 3, 1, 4],
                                      [5, 2, 3, 1],
                                      [3, 4, 2, 2],
                                      [2, 1, 5, 3]]> : tensor<4x4xi32>
  %in_i1 = stablehlo.constant dense<[[true,  false, true,  false],
                                     [true,  false, false, false],
                                     [true,  false, true,  true],
                                     [true,  false, true,  false]]> : tensor<4x4xi1>
  %in_f32 = stablehlo.constant dense<[[0.6, -0.5, 2.0, 0.2],
                                      [0.3, -1.0, 3.0, 1.5],
                                      [0.5,  1.2, 0.4, 0.8],
                                      [1.8, -0.2, 2.5, 0.7]]> : tensor<4x4xf32>

  %seq:8 = func.call @sequential_reduce_non_standard(%in_i32, %in_i1, %in_f32)
    : (tensor<4x4xi32>, tensor<4x4xi1>, tensor<4x4xf32>)
    -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi1>, tensor<4xi1>, tensor<4xi32>, tensor<4xf32>, tensor<4xi32>, tensor<4xi32>)

  %i32_0 = stablehlo.slice %in_i32 [0:2, 0:4] : (tensor<4x4xi32>) -> tensor<2x4xi32>
  %i32_1 = stablehlo.slice %in_i32 [2:4, 0:4] : (tensor<4x4xi32>) -> tensor<2x4xi32>
  %i1_0 = stablehlo.slice %in_i1 [0:2, 0:4] : (tensor<4x4xi1>) -> tensor<2x4xi1>
  %i1_1 = stablehlo.slice %in_i1 [2:4, 0:4] : (tensor<4x4xi1>) -> tensor<2x4xi1>
  %f32_0 = stablehlo.slice %in_f32 [0:2, 0:4] : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %f32_1 = stablehlo.slice %in_f32 [2:4, 0:4] : (tensor<4x4xf32>) -> tensor<2x4xf32>

  %res:16 = "interpreter.run_parallel"(%i32_0, %i1_0, %f32_0, %i32_1, %i1_1, %f32_1) {
    programs = [[@parallel_reduce_non_standard, @parallel_reduce_non_standard]]
  } : (tensor<2x4xi32>, tensor<2x4xi1>, tensor<2x4xf32>,
       tensor<2x4xi32>, tensor<2x4xi1>, tensor<2x4xf32>)
    -> (tensor<4xi32>, tensor<2xi32>, tensor<4xi1>, tensor<2xi1>, tensor<4xi32>, tensor<2xf32>, tensor<4xi32>, tensor<4xi32>,
        tensor<4xi32>, tensor<2xi32>, tensor<4xi1>, tensor<2xi1>, tensor<4xi32>, tensor<2xf32>, tensor<4xi32>, tensor<4xi32>)

  "check.expect_eq"(%res#0, %seq#0) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res#8, %seq#0) : (tensor<4xi32>, tensor<4xi32>) -> ()

  %prod_sharded_full = stablehlo.concatenate %res#1, %res#9, dim = 0 : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>
  "check.expect_eq"(%prod_sharded_full, %seq#1) : (tensor<4xi32>, tensor<4xi32>) -> ()

  "check.expect_eq"(%res#2, %seq#2) : (tensor<4xi1>, tensor<4xi1>) -> ()
  "check.expect_eq"(%res#10, %seq#2) : (tensor<4xi1>, tensor<4xi1>) -> ()

  %any_sharded_full = stablehlo.concatenate %res#3, %res#11, dim = 0 : (tensor<2xi1>, tensor<2xi1>) -> tensor<4xi1>
  "check.expect_eq"(%any_sharded_full, %seq#3) : (tensor<4xi1>, tensor<4xi1>) -> ()

  "check.expect_eq"(%res#4, %seq#4) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res#12, %seq#4) : (tensor<4xi32>, tensor<4xi32>) -> ()

  %custom_sharded_full = stablehlo.concatenate %res#5, %res#13, dim = 0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
  "check.expect_eq"(%custom_sharded_full, %seq#5) : (tensor<4xf32>, tensor<4xf32>) -> ()

  "check.expect_eq"(%res#6, %seq#6) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res#14, %seq#6) : (tensor<4xi32>, tensor<4xi32>) -> ()

  "check.expect_eq"(%res#7, %seq#7) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res#15, %seq#7) : (tensor<4xi32>, tensor<4xi32>) -> ()

  return
}
