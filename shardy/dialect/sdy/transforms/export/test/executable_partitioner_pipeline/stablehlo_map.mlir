// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh = <["x"=2]>

func.func @parallel_map(
  %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
  %arg1: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
  %arg2: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
  %arg3: tensor<5x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
  %arg4: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>})
  -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
      tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // Case 1: Matching sharded operands with compound scalar computation.
  %mapped_matching = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%x: tensor<f32>, %y: tensor<f32>):
      %scale = stablehlo.constant dense<2.500000e+00> : tensor<f32>
      %prod = stablehlo.multiply %x, %y : tensor<f32>
      %shifted = stablehlo.add %prod, %scale : tensor<f32>
      stablehlo.return %shifted : tensor<f32>
  }) {dimensions = array<i64: 0, 1>, sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

  // Case 2: Mismatched operand shardings requiring explicit reshard (all-to-all).
  %mapped_resharded = "stablehlo.map"(%arg0, %arg2) ({
    ^bb0(%x: tensor<f32>, %y: tensor<f32>):
      %diff = stablehlo.subtract %x, %y : tensor<f32>
      %abs_diff = stablehlo.abs %diff : tensor<f32>
      stablehlo.return %abs_diff : tensor<f32>
  }) {dimensions = array<i64: 0, 1>, sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

  // Case 3: Indivisible sharded map (5x4 on x=2) followed by reduction along dim 0.
  %arg3_sharded = sdy.reshard %arg3 <@mesh, [{"x"}, {}]> : tensor<5x4xf32>
  %mapped_indivisible = "stablehlo.map"(%arg3_sharded) ({
    ^bb0(%x: tensor<f32>):
      %one = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %inc = stablehlo.add %x, %one : tensor<f32>
      stablehlo.return %inc : tensor<f32>
  }) {dimensions = array<i64: 0, 1>, sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>} : (tensor<5x4xf32>) -> tensor<5x4xf32>
  %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %reduced = stablehlo.reduce(%mapped_indivisible init: %c0) applies stablehlo.add across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}]>]>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<4xf32>

  // Case 4: Sharded map capturing an outer scalar value inside its computation.
  %mapped_captured = "stablehlo.map"(%arg0) ({
    ^bb0(%x: tensor<f32>):
      %biased = stablehlo.add %x, %arg4 : tensor<f32>
      stablehlo.return %biased : tensor<f32>
  }) {dimensions = array<i64: 0, 1>, sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>} : (tensor<4x4xf32>) -> tensor<4x4xf32>

  return %mapped_matching, %mapped_resharded, %reduced, %mapped_captured
    : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>
}

func.func @sequential_map(
  %arg0: tensor<4x4xf32>,
  %arg1: tensor<4x4xf32>,
  %arg2: tensor<4x4xf32>,
  %arg3: tensor<5x4xf32>,
  %arg4: tensor<f32>)
  -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>) {
  %mapped_matching = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%x: tensor<f32>, %y: tensor<f32>):
      %scale = stablehlo.constant dense<2.500000e+00> : tensor<f32>
      %prod = stablehlo.multiply %x, %y : tensor<f32>
      %shifted = stablehlo.add %prod, %scale : tensor<f32>
      stablehlo.return %shifted : tensor<f32>
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

  %mapped_resharded = "stablehlo.map"(%arg0, %arg2) ({
    ^bb0(%x: tensor<f32>, %y: tensor<f32>):
      %diff = stablehlo.subtract %x, %y : tensor<f32>
      %abs_diff = stablehlo.abs %diff : tensor<f32>
      stablehlo.return %abs_diff : tensor<f32>
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

  %mapped_indivisible = "stablehlo.map"(%arg3) ({
    ^bb0(%x: tensor<f32>):
      %one = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %inc = stablehlo.add %x, %one : tensor<f32>
      stablehlo.return %inc : tensor<f32>
  }) {dimensions = array<i64: 0, 1>} : (tensor<5x4xf32>) -> tensor<5x4xf32>
  %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %reduced = stablehlo.reduce(%mapped_indivisible init: %c0) applies stablehlo.add across dimensions = [0]
    : (tensor<5x4xf32>, tensor<f32>) -> tensor<4xf32>

  %mapped_captured = "stablehlo.map"(%arg0) ({
    ^bb0(%x: tensor<f32>):
      %biased = stablehlo.add %x, %arg4 : tensor<f32>
      stablehlo.return %biased : tensor<f32>
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x4xf32>) -> tensor<4x4xf32>

  return %mapped_matching, %mapped_resharded, %reduced, %mapped_captured
    : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>
}

//--- part2.mlir

func.func @main() {
  %in0 = stablehlo.constant dense<[[1.0, -2.0, 3.0, 4.0],
                                   [5.0, 6.0, -7.0, 8.0],
                                   [-1.0, 2.0, 0.5, 1.5],
                                   [3.5, -4.5, 2.5, 0.0]]> : tensor<4x4xf32>
  %in1 = stablehlo.constant dense<[[0.5, 1.5, -2.0, 3.0],
                                   [2.0, -1.0, 4.0, 0.5],
                                   [3.0, 2.0, 1.0, -2.0],
                                   [-0.5, 1.0, 2.0, 5.0]]> : tensor<4x4xf32>
  %in2 = stablehlo.constant dense<[[10.0, 20.0, 30.0, 40.0],
                                   [1.0, 2.0, 3.0, 4.0],
                                   [5.0, 6.0, 7.0, 8.0],
                                   [-2.0, -4.0, -6.0, -8.0]]> : tensor<4x4xf32>
  %in3 = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0],
                                   [5.0, 6.0, 7.0, 8.0],
                                   [9.0, 10.0, 11.0, 12.0],
                                   [13.0, 14.0, 15.0, 16.0],
                                   [17.0, 18.0, 19.0, 20.0]]> : tensor<5x4xf32>
  %in4 = stablehlo.constant dense<-3.25> : tensor<f32>

  %seq:4 = func.call @sequential_map(%in0, %in1, %in2, %in3, %in4)
    : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<5x4xf32>, tensor<f32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>)

  %in0_0 = stablehlo.slice %in0 [0:2, 0:4] : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %in0_1 = stablehlo.slice %in0 [2:4, 0:4] : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %in1_0 = stablehlo.slice %in1 [0:2, 0:4] : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %in1_1 = stablehlo.slice %in1 [2:4, 0:4] : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %in2_0 = stablehlo.slice %in2 [0:4, 0:2] : (tensor<4x4xf32>) -> tensor<4x2xf32>
  %in2_1 = stablehlo.slice %in2 [0:4, 2:4] : (tensor<4x4xf32>) -> tensor<4x2xf32>

  %res:8 = "interpreter.run_parallel"(%in0_0, %in1_0, %in2_0, %in3, %in4, %in0_1, %in1_1, %in2_1, %in3, %in4) {
    programs = [[@parallel_map, @parallel_map]]
  } : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<4x2xf32>, tensor<5x4xf32>, tensor<f32>,
       tensor<2x4xf32>, tensor<2x4xf32>, tensor<4x2xf32>, tensor<5x4xf32>, tensor<f32>)
    -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<2x4xf32>,
        tensor<2x4xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<2x4xf32>)

  %matching_full = stablehlo.concatenate %res#0, %res#4, dim = 0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  "check.expect_eq"(%matching_full, %seq#0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()

  %resharded_full = stablehlo.concatenate %res#1, %res#5, dim = 0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  "check.expect_eq"(%resharded_full, %seq#1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()

  "check.expect_eq"(%res#2, %seq#2) : (tensor<4xf32>, tensor<4xf32>) -> ()
  "check.expect_eq"(%res#6, %seq#2) : (tensor<4xf32>, tensor<4xf32>) -> ()

  %captured_full = stablehlo.concatenate %res#3, %res#7, dim = 0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  "check.expect_eq"(%captured_full, %seq#3) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()

  return
}
