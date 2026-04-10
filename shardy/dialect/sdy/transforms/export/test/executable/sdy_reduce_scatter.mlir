// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

func.func @per_dim_reduce_scatter(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}]>}
) -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>}) {
  %0 = sdy.reduce_scatter [{"x"}, {}] %arg0 out_sharding=<@mesh_2_2, [{"x"}, {}]> : tensor<4x4xi32>
  %1 = sdy.reduce_scatter [{}, {"y"}] %0 out_sharding=<@mesh_2_2, [{"x"}, {"y"}]> : tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

func.func @combined_reduce_scatter(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}]>}
) -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>}) {
  %0 = sdy.reduce_scatter [{"x"}, {"y"}] %arg0 out_sharding=<@mesh_2_2, [{"x"}, {"y"}]> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

//--- part2.mlir

func.func @main() {
  %in0 = stablehlo.iota dim = 0 : tensor<4x4xi32>

  // Create distinct inputs for each of the 4 devices using multipliers.
  %c10 = stablehlo.constant dense<10> : tensor<4x4xi32>
  %c100 = stablehlo.constant dense<100> : tensor<4x4xi32>
  %c1000 = stablehlo.constant dense<1000> : tensor<4x4xi32>

  %in1 = stablehlo.multiply %in0, %c10 : tensor<4x4xi32>
  %in2 = stablehlo.multiply %in0, %c100 : tensor<4x4xi32>
  %in3 = stablehlo.multiply %in0, %c1000 : tensor<4x4xi32>

  %res_comb:4 = "interpreter.run_parallel"(%in0, %in1, %in2, %in3) {
    programs = [[@combined_reduce_scatter, @combined_reduce_scatter, @combined_reduce_scatter, @combined_reduce_scatter]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) ->
      (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>)

  %res_seq:4 = "interpreter.run_parallel"(%in0, %in1, %in2, %in3) {
    programs = [[@per_dim_reduce_scatter, @per_dim_reduce_scatter, @per_dim_reduce_scatter, @per_dim_reduce_scatter]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) ->
      (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>)

  "check.expect_eq"(%res_seq#0, %res_comb#0) : (tensor<2x2xi32>, tensor<2x2xi32>) -> ()
  "check.expect_eq"(%res_seq#1, %res_comb#1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> ()
  "check.expect_eq"(%res_seq#2, %res_comb#2) : (tensor<2x2xi32>, tensor<2x2xi32>) -> ()
  "check.expect_eq"(%res_seq#3, %res_comb#3) : (tensor<2x2xi32>, tensor<2x2xi32>) -> ()

  return
}


