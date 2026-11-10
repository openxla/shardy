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
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  } : (tensor<4x2xf32>, tensor<2x1xi64>, tensor<2x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// ([i, j], [k, v], [k, j]) -> ([i, j]) reduction={k}
//
func.func @parallel_scatter(
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>},
  %arg1: tensor<2x1xi64> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg2: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> tensor<4x2xf32> {

  %pid = stablehlo.partition_id : tensor<ui32>

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
    >
  } : (tensor<4x2xf32>, tensor<2x1xi64>, tensor<2x2xf32>) -> tensor<4x2xf32>

  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<4x2xf32>
  return %1 : tensor<4x2xf32>
}

func.func @sequential_scatter_unsigned_min(
  %arg0: tensor<4x2xui32>,
  %arg1: tensor<2x1xui8>,
  %arg2: tensor<2x2xui32>) -> tensor<4x2xui32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<ui32>, %arg4: tensor<ui32>):
      %1 = stablehlo.minimum %arg3, %arg4 : tensor<ui32>
      stablehlo.return %1 : tensor<ui32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  } : (tensor<4x2xui32>, tensor<2x1xui8>, tensor<2x2xui32>) -> tensor<4x2xui32>
  return %0 : tensor<4x2xui32>
}

func.func @parallel_scatter_unsigned_min(
  %arg0: tensor<4x2xui32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>},
  %arg1: tensor<2x1xui8> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg2: tensor<2x2xui32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<4x2xui32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<ui32>, %arg4: tensor<ui32>):
      %1 = stablehlo.minimum %arg3, %arg4 : tensor<ui32>
      stablehlo.return %1 : tensor<ui32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}]>]>
  } : (tensor<4x2xui32>, tensor<2x1xui8>, tensor<2x2xui32>) -> tensor<4x2xui32>
  return %0 : tensor<4x2xui32>
}

func.func @sequential_scatter_sharded_operand_narrow_indices(
  %arg0: tensor<200x2xui32>,
  %arg1: tensor<2x1xi8>,
  %arg2: tensor<2x2xui32>) -> tensor<200x2xui32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<ui32>, %arg4: tensor<ui32>):
      %1 = stablehlo.maximum %arg3, %arg4 : tensor<ui32>
      stablehlo.return %1 : tensor<ui32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >
  } : (tensor<200x2xui32>, tensor<2x1xi8>, tensor<2x2xui32>) -> tensor<200x2xui32>
  return %0 : tensor<200x2xui32>
}

func.func @parallel_scatter_sharded_operand_narrow_indices(
  %arg0: tensor<200x2xui32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2x1xi8> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>},
  %arg2: tensor<2x2xui32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>})
  -> (tensor<200x2xui32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<ui32>, %arg4: tensor<ui32>):
      %1 = stablehlo.maximum %arg3, %arg4 : tensor<ui32>
      stablehlo.return %1 : tensor<ui32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<200x2xui32>, tensor<2x1xi8>, tensor<2x2xui32>) -> tensor<200x2xui32>
  return %0 : tensor<200x2xui32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<1.0> : tensor<4x2xf32>
  %indices = stablehlo.constant dense<[[1], [3]]> : tensor<2x1xi64>
  %updates = stablehlo.constant dense<[[10.0, 20.0], [30.0, 40.0]]> : tensor<2x2xf32>

  %seq = func.call @sequential_scatter(%input, %indices, %updates)
    : (tensor<4x2xf32>, tensor<2x1xi64>, tensor<2x2xf32>) -> tensor<4x2xf32>

  %idx0 = "stablehlo.slice"(%indices) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 1,1>, strides=array<i64: 1,1>} : (tensor<2x1xi64>) -> tensor<1x1xi64>
  %upd0 = "stablehlo.slice"(%updates) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 1,2>, strides=array<i64: 1,1>} : (tensor<2x2xf32>) -> tensor<1x2xf32>
  %idx1 = "stablehlo.slice"(%indices) {start_indices=array<i64: 1,0>, limit_indices=array<i64: 2,1>, strides=array<i64: 1,1>} : (tensor<2x1xi64>) -> tensor<1x1xi64>
  %upd1 = "stablehlo.slice"(%updates) {start_indices=array<i64: 1,0>, limit_indices=array<i64: 2,2>, strides=array<i64: 1,1>} : (tensor<2x2xf32>) -> tensor<1x2xf32>

  %pars:2 = "interpreter.run_parallel"(%input, %idx0, %upd0, %input, %idx1, %upd1) {
    programs = [[@parallel_scatter, @parallel_scatter]]
  } : (tensor<4x2xf32>, tensor<1x1xi64>, tensor<1x2xf32>, tensor<4x2xf32>, tensor<1x1xi64>, tensor<1x2xf32>)
    -> (tensor<4x2xf32>, tensor<4x2xf32>)

  "check.expect_eq"(%pars#0, %seq) : (tensor<4x2xf32>, tensor<4x2xf32>) -> ()
  "check.expect_eq"(%pars#1, %seq) : (tensor<4x2xf32>, tensor<4x2xf32>) -> ()

  %in_ui32 = stablehlo.constant dense<3000000000> : tensor<4x2xui32>
  %idx_ui8 = stablehlo.constant dense<[[1], [3]]> : tensor<2x1xui8>
  %upd_ui32 = stablehlo.constant dense<[[10, 20], [30, 40]]> : tensor<2x2xui32>
  %seq_umin = func.call @sequential_scatter_unsigned_min(%in_ui32, %idx_ui8, %upd_ui32)
    : (tensor<4x2xui32>, tensor<2x1xui8>, tensor<2x2xui32>) -> tensor<4x2xui32>
  %idx_ui8_0 = "stablehlo.slice"(%idx_ui8) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 1,1>, strides=array<i64: 1,1>} : (tensor<2x1xui8>) -> tensor<1x1xui8>
  %upd_ui32_0 = "stablehlo.slice"(%upd_ui32) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 1,2>, strides=array<i64: 1,1>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
  %idx_ui8_1 = "stablehlo.slice"(%idx_ui8) {start_indices=array<i64: 1,0>, limit_indices=array<i64: 2,1>, strides=array<i64: 1,1>} : (tensor<2x1xui8>) -> tensor<1x1xui8>
  %upd_ui32_1 = "stablehlo.slice"(%upd_ui32) {start_indices=array<i64: 1,0>, limit_indices=array<i64: 2,2>, strides=array<i64: 1,1>} : (tensor<2x2xui32>) -> tensor<1x2xui32>
  %pars_umin:2 = "interpreter.run_parallel"(%in_ui32, %idx_ui8_0, %upd_ui32_0, %in_ui32, %idx_ui8_1, %upd_ui32_1) {
    programs = [[@parallel_scatter_unsigned_min, @parallel_scatter_unsigned_min]]
  } : (tensor<4x2xui32>, tensor<1x1xui8>, tensor<1x2xui32>, tensor<4x2xui32>, tensor<1x1xui8>, tensor<1x2xui32>)
    -> (tensor<4x2xui32>, tensor<4x2xui32>)
  "check.expect_eq"(%pars_umin#0, %seq_umin) : (tensor<4x2xui32>, tensor<4x2xui32>) -> ()
  "check.expect_eq"(%pars_umin#1, %seq_umin) : (tensor<4x2xui32>, tensor<4x2xui32>) -> ()

  %in_200 = stablehlo.constant dense<5> : tensor<200x2xui32>
  %idx_i8 = stablehlo.constant dense<[[10], [110]]> : tensor<2x1xi8>
  %seq_narrow = func.call @sequential_scatter_sharded_operand_narrow_indices(%in_200, %idx_i8, %upd_ui32)
    : (tensor<200x2xui32>, tensor<2x1xi8>, tensor<2x2xui32>) -> tensor<200x2xui32>
  %in_200_0 = "stablehlo.slice"(%in_200) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 100,2>, strides=array<i64: 1,1>} : (tensor<200x2xui32>) -> tensor<100x2xui32>
  %in_200_1 = "stablehlo.slice"(%in_200) {start_indices=array<i64: 100,0>, limit_indices=array<i64: 200,2>, strides=array<i64: 1,1>} : (tensor<200x2xui32>) -> tensor<100x2xui32>
  %pars_narrow:2 = "interpreter.run_parallel"(%in_200_0, %idx_i8, %upd_ui32, %in_200_1, %idx_i8, %upd_ui32) {
    programs = [[@parallel_scatter_sharded_operand_narrow_indices, @parallel_scatter_sharded_operand_narrow_indices]]
  } : (tensor<100x2xui32>, tensor<2x1xi8>, tensor<2x2xui32>, tensor<100x2xui32>, tensor<2x1xi8>, tensor<2x2xui32>)
    -> (tensor<100x2xui32>, tensor<100x2xui32>)
  %par_narrow = stablehlo.concatenate %pars_narrow#0, %pars_narrow#1, dim = 0
    : (tensor<100x2xui32>, tensor<100x2xui32>) -> tensor<200x2xui32>
  "check.expect_eq"(%par_narrow, %seq_narrow) : (tensor<200x2xui32>, tensor<200x2xui32>) -> ()
  return
}
