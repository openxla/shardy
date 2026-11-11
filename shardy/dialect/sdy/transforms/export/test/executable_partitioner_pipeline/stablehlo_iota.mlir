// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

func.func @parallel_iota_1d()
    -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : tensor<4xi32>
  return %0 : tensor<4xi32>
}

func.func @parallel_iota_2d_dim0()
    -> (tensor<4x6xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}]>]>} : tensor<4x6xi32>
  return %0 : tensor<4x6xi32>
}

func.func @parallel_iota_2d_dim1()
    -> (tensor<4x6xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}]>]>} : tensor<4x6xi32>
  return %0 : tensor<4x6xi32>
}

//--- part2.mlir

func.func @main() {
  // Test 1: 1D mesh sharded along iota dimension 0.
  %seq_1d = func.call @sequential_iota_1d() : () -> tensor<4xi32>

  %pars_1d:2 = "interpreter.run_parallel"() {
    programs = [[@parallel_iota_1d, @parallel_iota_1d]]
  } : () -> (tensor<2xi32>, tensor<2xi32>)

  %par_1d = "stablehlo.concatenate"(%pars_1d#0, %pars_1d#1) {
    dimension = 0 : i64
  } : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>

  check.expect_eq %seq_1d, %par_1d : tensor<4xi32>

  // Test 2: 2D mesh sharded on iota dimension 0 ("x"=2) and non-iota dimension 1 ("y"=2).
  %seq_2d_dim0 = func.call @sequential_iota_2d_dim0() : () -> tensor<4x6xi32>

  %pars_2d_dim0:4 = "interpreter.run_parallel"() {
    programs = [[@parallel_iota_2d_dim0, @parallel_iota_2d_dim0, @parallel_iota_2d_dim0, @parallel_iota_2d_dim0]]
  } : () -> (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>)

  %row01_dim0 = "stablehlo.concatenate"(%pars_2d_dim0#0, %pars_2d_dim0#1) {
    dimension = 1 : i64
  } : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x6xi32>
  %row23_dim0 = "stablehlo.concatenate"(%pars_2d_dim0#2, %pars_2d_dim0#3) {
    dimension = 1 : i64
  } : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x6xi32>
  %par_2d_dim0 = "stablehlo.concatenate"(%row01_dim0, %row23_dim0) {
    dimension = 0 : i64
  } : (tensor<2x6xi32>, tensor<2x6xi32>) -> tensor<4x6xi32>

  check.expect_eq %seq_2d_dim0, %par_2d_dim0 : tensor<4x6xi32>

  // Test 3: 2D mesh sharded on non-iota dimension 0 ("x"=2) and iota dimension 1 ("y"=2).
  %seq_2d_dim1 = func.call @sequential_iota_2d_dim1() : () -> tensor<4x6xi32>

  %pars_2d_dim1:4 = "interpreter.run_parallel"() {
    programs = [[@parallel_iota_2d_dim1, @parallel_iota_2d_dim1, @parallel_iota_2d_dim1, @parallel_iota_2d_dim1]]
  } : () -> (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>)

  %row01_dim1 = "stablehlo.concatenate"(%pars_2d_dim1#0, %pars_2d_dim1#1) {
    dimension = 1 : i64
  } : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x6xi32>
  %row23_dim1 = "stablehlo.concatenate"(%pars_2d_dim1#2, %pars_2d_dim1#3) {
    dimension = 1 : i64
  } : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x6xi32>
  %par_2d_dim1 = "stablehlo.concatenate"(%row01_dim1, %row23_dim1) {
    dimension = 0 : i64
  } : (tensor<2x6xi32>, tensor<2x6xi32>) -> tensor<4x6xi32>

  check.expect_eq %seq_2d_dim1, %par_2d_dim1 : tensor<4x6xi32>

  return
}
