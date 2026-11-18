// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_xy = <["x"=2, "y"=2]>

func.func @parallel_reshape_non_equiv_divisible(
  %arg0: tensor<3x8x40xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}, {"x"}]>})
  -> (tensor<2x480xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {}]>}) {
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_xy, [{}, {}]>]>} : (tensor<3x8x40xi32>) -> tensor<2x480xi32>
  return %0 : tensor<2x480xi32>
}

func.func @sequential_reshape_non_equiv_divisible(%arg0: tensor<3x8x40xi32>) -> tensor<2x480xi32> {
  %0 = stablehlo.reshape %arg0 : (tensor<3x8x40xi32>) -> tensor<2x480xi32>
  return %0 : tensor<2x480xi32>
}

//--- part2.mlir

func.func @main() {
  %iota_1d = stablehlo.iota dim = 0 : tensor<960xi32>
  %c42 = stablehlo.constant dense<42> : tensor<960xi32>
  %sum_1d = stablehlo.add %iota_1d, %c42 : tensor<960xi32>
  %input = stablehlo.reshape %sum_1d : (tensor<960xi32>) -> tensor<3x8x40xi32>
  %seq = func.call @sequential_reshape_non_equiv_divisible(%input) : (tensor<3x8x40xi32>) -> tensor<2x480xi32>

  %t0 = stablehlo.slice %input [0:3, 0:4, 0:20] : (tensor<3x8x40xi32>) -> tensor<3x4x20xi32>
  %t1 = stablehlo.slice %input [0:3, 4:8, 0:20] : (tensor<3x8x40xi32>) -> tensor<3x4x20xi32>
  %t2 = stablehlo.slice %input [0:3, 0:4, 20:40] : (tensor<3x8x40xi32>) -> tensor<3x4x20xi32>
  %t3 = stablehlo.slice %input [0:3, 4:8, 20:40] : (tensor<3x8x40xi32>) -> tensor<3x4x20xi32>

  %res:4 = "interpreter.run_parallel"(%t0, %t1, %t2, %t3) {
    programs = [[@parallel_reshape_non_equiv_divisible, @parallel_reshape_non_equiv_divisible, @parallel_reshape_non_equiv_divisible, @parallel_reshape_non_equiv_divisible]]
  } : (tensor<3x4x20xi32>, tensor<3x4x20xi32>, tensor<3x4x20xi32>, tensor<3x4x20xi32>) -> (tensor<2x480xi32>, tensor<2x480xi32>, tensor<2x480xi32>, tensor<2x480xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<2x480xi32>, tensor<2x480xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<2x480xi32>, tensor<2x480xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<2x480xi32>, tensor<2x480xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<2x480xi32>, tensor<2x480xi32>) -> ()

  return
}
