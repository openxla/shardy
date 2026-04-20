// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @sequential_scatter(
  %arg0: tensor<4x2xf32>,
  %arg1: tensor<2x1xi64>,
  %arg2: tensor<2x2xf32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0], // Row dimension is collapsed/inserted
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  } : (tensor<4x2xf32>, tensor<2x1xi64>, tensor<2x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// ([b, w], [b, k, v], [b, k, w]) -> ([b, w]) reduction={k}
//
func.func @parallel_scatter(
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2x1xi64>,
  %arg2: tensor<2x2xf32>)
  -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {

  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<4x2xf32>, tensor<2x1xi64>, tensor<2x2xf32>) -> tensor<4x2xf32>

  return %0 : tensor<4x2xf32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<1.0> : tensor<4x2xf32>
  %indices = stablehlo.constant dense<[[1], [3]]> : tensor<2x1xi64>
  %updates = stablehlo.constant dense<[[10.0, 20.0], [30.0, 40.0]]> : tensor<2x2xf32>

  %seq = func.call @sequential_scatter(%input, %indices, %updates)
    : (tensor<4x2xf32>, tensor<2x1xi64>, tensor<2x2xf32>) -> tensor<4x2xf32>

  // TODO(b/500779239): Remove this once the bug is fixed.
  //
  // The interpreter currently has an unexpected side effect of modifying the
  // input tensor. To work around this, we define a new input tensor for the
  // parallel run.
  %input_par = stablehlo.constant dense<1.0> : tensor<4x2xf32>

  %s0 = "stablehlo.slice"(%input_par) {
    start_indices = array<i64: 0,0>, limit_indices = array<i64: 2,2>, strides = array<i64: 1,1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %s1 = "stablehlo.slice"(%input_par) {
    start_indices = array<i64: 2,0>, limit_indices = array<i64: 4,2>, strides = array<i64: 1,1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>

  %pars:2 = "interpreter.run_parallel"(%s0, %indices, %updates, %s1, %indices, %updates) {
    programs = [[@parallel_scatter, @parallel_scatter]]
  } : (tensor<2x2xf32>, tensor<2x1xi64>, tensor<2x2xf32>,
       tensor<2x2xf32>, tensor<2x1xi64>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  %par = "stablehlo.concatenate"(%pars#0, %pars#1) { dimension = 0 : i64 }
    : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
  "check.expect_eq"(%seq, %par) : (tensor<4x2xf32>, tensor<4x2xf32>) -> ()
  return
}
