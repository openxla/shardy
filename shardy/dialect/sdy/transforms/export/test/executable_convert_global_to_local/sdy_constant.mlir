// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @sharded_dense_constant()
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} dense<[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
  ]> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

func.func @sharded_splat_constant()
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} dense<7> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

//--- part2.mlir

func.func @main() {
  %res_dense:2 = "interpreter.run_parallel"() {
    programs = [[@sharded_dense_constant, @sharded_dense_constant]]
  } : () -> (tensor<2x4xi32>, tensor<2x4xi32>)

  %e_dense_0 = stablehlo.constant dense<[
    [1, 2, 3, 4],
    [5, 6, 7, 8]
  ]> : tensor<2x4xi32>
  %e_dense_1 = stablehlo.constant dense<[
    [9, 10, 11, 12],
    [13, 14, 15, 16]
  ]> : tensor<2x4xi32>

  "check.expect_eq"(%res_dense#0, %e_dense_0) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
  "check.expect_eq"(%res_dense#1, %e_dense_1) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()

  %res_splat:2 = "interpreter.run_parallel"() {
    programs = [[@sharded_splat_constant, @sharded_splat_constant]]
  } : () -> (tensor<2x4xi32>, tensor<2x4xi32>)

  %e_splat = stablehlo.constant dense<7> : tensor<2x4xi32>

  "check.expect_eq"(%res_splat#0, %e_splat) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
  "check.expect_eq"(%res_splat#1, %e_splat) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()

  return
}
