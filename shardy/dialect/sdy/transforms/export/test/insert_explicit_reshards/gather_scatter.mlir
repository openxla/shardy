// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=8]>
sdy.mesh @mesh_xyztp = <["x"=4, "y"=4, "z"=4, "t"=8, "p"=4]>

// CHECK-LABEL: @gather
// COM: the most expressive example
func.func @gather(
  %arg0: tensor<2x6x4x26x22xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>},
  %arg1: tensor<2x22x12x26x14xi64> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x": (1)2}, {"z": (1)2}, {"z": (2)2}, {"y": (2)2}, {"t"}]>}
) -> (tensor<1x2x22x12x26x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {}, {"z": (1)2}, {"z": (2)2}, {"y": (2)2}, {"t"}]>}) {
  // COM: {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, o, m, k], [p, k, l, m, n])->([i, j, k, l, m, n]) {...} need_replication={i, j, p} blocked_propagation={i, j}>}

  // CHECK-DAG: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {}, {"y":(1)2}, {"y":(2)2}, {"z":(1)2}]> : tensor
  // CHECK-DAG: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyzt, [{}, {"z":(1)2}, {"z":(2)2}, {"y":(2)2}, {"t"}]> : tensor
  // CHECK-DAG: %[[GATHER:.*]] = "stablehlo.gather"(%[[RESHARD1]], %[[RESHARD2]])
  // CHECK-DAG: return %[[GATHER]] : tensor
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0, 1],
      collapsed_slice_dims = [2],
      operand_batching_dims = [3, 4],
      start_indices_batching_dims = [3, 1],
      start_index_map = [1, 0],
      index_vector_dim = 0>,
    slice_sizes = array<i64: 1, 2, 1, 1, 1>,
    indices_are_sorted = false,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {}, {"z": (1)2}, {"z": (2)2}, {"y": (2)2}, {"t"}]>]>
  } : (tensor<2x6x4x26x22xf32>, tensor<2x22x12x26x14xi64>) -> tensor<1x2x22x12x26x14xf32>
  return %0 : tensor<1x2x22x12x26x14xf32>
}

// CHECK-LABEL: @gather_implicit_dimension
func.func @gather_implicit_dimension(
  %arg0: tensor<2x6x4x26x22xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>},
  %arg1: tensor<22x12x26x14xi64> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"z": (1)2}, {"z": (2)2}, {"y": (2)2}, {"t"}]>}
) -> (tensor<1x2x22x12x26x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {}, {"z": (1)2}, {"z": (2)2}, {"y": (2)2}, {"t"}]>}) {
  // COM: #sdy.op_sharding_rule<([i, j, o, m, k], [k, l, m, n])->([i, j, k, l, m, n]) {...} need_replication={i, j} blocked_propagation={i, j}>

  // CHECK-DAG: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {}, {"y":(1)2}, {"y":(2)2}, {"z":(1)2}]>
  // CHECK-DAG: %[[GATHER:.*]] = "stablehlo.gather"(%[[RESHARD1]], %arg1)
  // CHECK-DAG: return %[[GATHER]]
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0, 1],
      collapsed_slice_dims = [2],
      operand_batching_dims = [3, 4],
      start_indices_batching_dims = [2, 0],
      start_index_map = [1],
      index_vector_dim = 4>,
    slice_sizes = array<i64: 1, 2, 1, 1, 1>,
    indices_are_sorted = false,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {}, {"z": (1)2}, {"z": (2)2}, {"y": (2)2}, {"t"}]>]>
  } : (tensor<2x6x4x26x22xf32>, tensor<22x12x26x14xi64>) -> tensor<1x2x22x12x26x14xf32>
  return %0 : tensor<1x2x22x12x26x14xf32>
}

// CHECK-LABEL: @gather_full_offset_dimension
func.func @gather_full_offset_dimension(
  %arg0: tensor<2x6x4x26x22xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>},
  %arg1: tensor<2x22x12x26x14xi64> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x": (1)2}, {"z": (1)2}, {"z": (2)2}, {"y": (2)2}, {"t"}]>}
) -> (tensor<2x6x22x12x26x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x": (1)2}, {"x": (2)2}, {"z": (1)2}, {"z": (2)2}, {"y": (2)2}, {"t"}]>}) {
  // COM: #sdy.op_sharding_rule<([i, j, o, m, k], [p, k, l, m, n])->([i, j, k, l, m, n]) {...} need_replication={p}>

  // CHECK-DAG: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyzt, [{}, {"z":(1)2}, {"z":(2)2}, {"y":(2)2}, {"t"}]> : tensor
  // CHECK-DAG: %[[GATHER:.*]] = "stablehlo.gather"(%arg0, %[[RESHARD2]])
  // CHECK-DAG: return %[[GATHER]] : tensor
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0, 1],
      collapsed_slice_dims = [2],
      operand_batching_dims = [3, 4],
      start_indices_batching_dims = [3, 1],
      start_index_map = [1, 0],
      index_vector_dim = 0>,
    slice_sizes = array<i64: 2, 6, 1, 1, 1>,
    indices_are_sorted = false,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"x":(1)2}, {"x":(2)2}, {"z": (1)2}, {"z": (2)2}, {"y": (2)2}, {"t"}]>]>
  } : (tensor<2x6x4x26x22xf32>, tensor<2x22x12x26x14xi64>) -> tensor<2x6x22x12x26x14xf32>
  return %0 : tensor<2x6x22x12x26x14xf32>
}


// CHECK-LABEL: @scatter
// COM: the most expressive example
func.func @scatter(
  %arg0: tensor<6x4x10x12x14xf32>     {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>},
  %arg1: tensor<12x22x4x2x26xi64>   {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"z": (2)2}, {"x": (2)2}, {"t": (1)2}, {"t": (2)2}]>},
  %arg2: tensor<12x22x2x4x26x10xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"z": (2)2}, {"t": (4)2}, {"x": (2)2}, {"t": (2)2}, {"p": (1)2}]>}
) -> (tensor<6x4x10x12x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>}){
  // COM: #sdy.op_sharding_rule<([k, m, q, i, o], [i, j, m, r, n], [i, j, l, m, n, p])->([k, m, q, i, o]) {...} reduction={j, n} need_replication={l, p, r}>

  // CHECK-DAG: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyztp, [{"y":(2)2}, {"z":(2)2}, {"x":(2)2}, {}, {"t":(2)2}]> : tensor
  // CHECK-DAG: %[[RESHARD2:.*]] = sdy.reshard %arg2 <@mesh_xyztp, [{"y":(2)2}, {"z":(2)2}, {}, {"x":(2)2}, {"t":(2)2}, {}]> : tensor
  // CHECK-DAG: %[[SCATTER:.*]] = "stablehlo.scatter"(%arg0, %[[RESHARD1]], %[[RESHARD2]])
  // CHECK-DAG: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"z":(2)2, "t":(2)2} %[[SCATTER]] out_sharding=<@mesh_xyztp, [{"x":(1)2}, {"x":(2)2}, {"y":(1)2}, {"y":(2)2}, {"z":(1)2}]>
  // CHECK-DAG: return %[[ALL_REDUCE]] : tensor
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 5],
      inserted_window_dims = [2],
      input_batching_dims = [1, 3],
      scatter_indices_batching_dims = [2, 0],
      scatter_dims_to_operand_dims = [2, 0],
      index_vector_dim = 3>,
    indices_are_sorted = false,
    unique_indices = false,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>]>
  } : (tensor<6x4x10x12x14xf32>, tensor<12x22x4x2x26xi64>, tensor<12x22x2x4x26x10xf32>) -> tensor<6x4x10x12x14xf32>
  return %0 : tensor<6x4x10x12x14xf32>
}

// CHECK-LABEL: @scatter_multi_inputs
func.func @scatter_multi_inputs(
  %arg0: tensor<6x4x10x12x14xf32>     {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>},
  %arg1: tensor<6x4x10x12x14xf32>     {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>},
  %arg2: tensor<12x22x4x2x26xi64>   {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"z": (2)2}, {"x": (2)2}, {"t": (1)2}, {"t": (2)2}]>},
  %arg3: tensor<12x22x2x4x26x10xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"z": (2)2}, {"t": (4)2}, {"x": (2)2}, {"t": (2)2}, {"p": (1)2}]>},
  %arg4: tensor<12x22x2x4x26x10xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"z": (2)2}, {"t": (4)2}, {"x": (2)2}, {"t": (2)2}, {"p": (1)2}]>}
) -> (tensor<6x4x10x12x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>}, tensor<6x4x10x12x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>}){
  // CHECK-DAG: %[[RESHARD1:.*]] = sdy.reshard %arg2 <@mesh_xyztp, [{"y":(2)2}, {"z":(2)2}, {"x":(2)2}, {}, {"t":(2)2}]> : tensor
  // CHECK-DAG: %[[RESHARD2:.*]] = sdy.reshard %arg3 <@mesh_xyztp, [{"y":(2)2}, {"z":(2)2}, {}, {"x":(2)2}, {"t":(2)2}, {}]> : tensor
  // CHECK-DAG: %[[RESHARD3:.*]] = sdy.reshard %arg4 <@mesh_xyztp, [{"y":(2)2}, {"z":(2)2}, {}, {"x":(2)2}, {"t":(2)2}, {}]> : tensor
  // CHECK-DAG: %[[SCATTER:.*]]:2 = "stablehlo.scatter"(%arg0, %arg1, %[[RESHARD1]], %[[RESHARD2]], %[[RESHARD3]])
  // CHECK-DAG: %[[ALL_REDUCE1:.*]] = sdy.all_reduce {"z":(2)2, "t":(2)2} %[[SCATTER]]#0 out_sharding=<@mesh_xyztp, [{"x":(1)2}, {"x":(2)2}, {"y":(1)2}, {"y":(2)2}, {"z":(1)2}]>
  // CHECK-DAG: %[[ALL_REDUCE2:.*]] = sdy.all_reduce {"z":(2)2, "t":(2)2} %[[SCATTER]]#1 out_sharding=<@mesh_xyztp, [{"x":(1)2}, {"x":(2)2}, {"y":(1)2}, {"y":(2)2}, {"z":(1)2}]>
  // CHECK-DAG: return %[[ALL_REDUCE1]], %[[ALL_REDUCE2]] : tensor
  %0:2 = "stablehlo.scatter"(%arg0, %arg1, %arg2, %arg3, %arg4) ({
    ^bb0(%arg11: tensor<f32>, %arg12: tensor<f32>, %arg13: tensor<f32>, %arg14: tensor<f32>):
      %1 = stablehlo.add %arg11, %arg12 : tensor<f32>
      %2 = stablehlo.add %arg13, %arg14 : tensor<f32>
      stablehlo.return %1, %2 : tensor<f32>, tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 5],
      inserted_window_dims = [2],
      input_batching_dims = [1, 3],
      scatter_indices_batching_dims = [2, 0],
      scatter_dims_to_operand_dims = [2, 0],
      index_vector_dim = 3>,
    indices_are_sorted = false,
    unique_indices = false,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>, <@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>]>
  } : (tensor<6x4x10x12x14xf32>, tensor<6x4x10x12x14xf32>, tensor<12x22x4x2x26xi64>, tensor<12x22x2x4x26x10xf32>, tensor<12x22x2x4x26x10xf32>) -> (tensor<6x4x10x12x14xf32>, tensor<6x4x10x12x14xf32>)
  return %0#0, %0#1 : tensor<6x4x10x12x14xf32>, tensor<6x4x10x12x14xf32>
}

// CHECK-LABEL: @scatter_implicit_dimension
func.func @scatter_implicit_dimension(
  %arg0: tensor<6x4x10x12x14xf32>     {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>},
  %arg1: tensor<12x22x4x26xi64>   {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"z": (2)2}, {"x": (2)2}, {"t": (2)2}]>},
  %arg2: tensor<12x22x2x4x26x10xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"z": (2)2}, {"t": (4)2}, {"x": (2)2}, {"t": (2)2}, {"p": (1)2}]>}
) -> (tensor<6x4x10x12x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>}){
  // COM: #sdy.op_sharding_rule<([k, m, q, i, o], [i, j, m, n], [i, j, l, m, n, p])->([k, m, q, i, o]) {...} reduction={j, n} need_replication={l, p}>

  // CHECK-DAG: %[[RESHARD1:.*]] = sdy.reshard %arg2 <@mesh_xyztp, [{"y":(2)2}, {"z":(2)2}, {}, {"x":(2)2}, {"t":(2)2}, {}]> : tensor
  // CHECK-DAG: %[[SCATTER:.*]] = "stablehlo.scatter"(%arg0, %arg1, %[[RESHARD1]])
  // CHECK-DAG: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"z":(2)2, "t":(2)2} %[[SCATTER]] out_sharding=<@mesh_xyztp, [{"x":(1)2}, {"x":(2)2}, {"y":(1)2}, {"y":(2)2}, {"z":(1)2}]>
  // CHECK-DAG: return %[[ALL_REDUCE]] : tensor
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 5],
      inserted_window_dims = [2],
      input_batching_dims = [1, 3],
      scatter_indices_batching_dims = [2, 0],
      scatter_dims_to_operand_dims = [2],
      index_vector_dim = 4>,
    indices_are_sorted = false,
    unique_indices = false,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>]>
  } : (tensor<6x4x10x12x14xf32>, tensor<12x22x4x26xi64>, tensor<12x22x2x4x26x10xf32>) -> tensor<6x4x10x12x14xf32>
  return %0 : tensor<6x4x10x12x14xf32>
}

// CHECK-LABEL: @scatter_full_update_dimensions
func.func @scatter_full_update_dimensions(
  %arg0: tensor<6x4x10x12x14xf32>     {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>},
  %arg1: tensor<12x22x4x2x26xi64>   {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"z": (2)2}, {"x": (2)2}, {"t": (1)2}, {"t": (2)2}]>},
  %arg2: tensor<12x22x6x4x26x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"z": (2)2}, {"x": (1)2}, {"x": (2)2}, {"t": (2)2}, {"z": (1)2}]>}
) -> (tensor<6x4x10x12x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>}){
  // COM: #sdy.op_sharding_rule<([k, l, o, i, n], [i, j, l, p, m], [i, j, k, l, m, n])->([k, l, o, i, n]) {i=12, j=22, k=6, l=4, m=26, n=14, o=10, p=2} reduction={j, m} need_replication={p}>

  // CHECK-DAG: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyztp, [{"y":(2)2}, {"z":(2)2}, {"x":(2)2}, {}, {"t":(2)2}]> : tensor
  // CHECK-DAG: %[[SCATTER:.*]] = "stablehlo.scatter"(%arg0, %[[RESHARD1]], %arg2)
  // CHECK-DAG: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"z":(2)2, "t":(2)2} %[[SCATTER]] out_sharding=<@mesh_xyztp, [{"x":(1)2}, {"x":(2)2}, {"y":(1)2}, {"y":(2)2}, {"z":(1)2}]>
  // CHECK-DAG: return %[[ALL_REDUCE]] : tensor
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 5],
      inserted_window_dims = [2],
      input_batching_dims = [1, 3],
      scatter_indices_batching_dims = [2, 0],
      scatter_dims_to_operand_dims = [2, 0],
      index_vector_dim = 3>,
    indices_are_sorted = false,
    unique_indices = false,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>]>
  } : (tensor<6x4x10x12x14xf32>, tensor<12x22x4x2x26xi64>, tensor<12x22x6x4x26x14xf32>) -> tensor<6x4x10x12x14xf32>
  return %0 : tensor<6x4x10x12x14xf32>
}

// CHECK-LABEL: @scatter_no_reduction
func.func @scatter_no_reduction(
  %arg0: tensor<6x4x10x12x14xf32>     {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>},
  %arg1: tensor<12x4x2xi64>   {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"x": (2)2}, {"t": (1)2}]>},
  %arg2: tensor<12x2x4x10xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"y": (2)2}, {"t": (4)2}, {"x": (2)2}, {"p": (1)2}]>}
) -> (tensor<6x4x10x12x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>}){
  // COM: #sdy.op_sharding_rule<([k, m, q, i, o], [i, j, m, r, n], [i, j, l, m, n, p])->([k, m, q, i, o]) {...} reduction={j, n} need_replication={l, p, r}>

  // CHECK-DAG: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyztp, [{"y":(2)2}, {"x":(2)2}, {}]> : tensor
  // CHECK-DAG: %[[RESHARD2:.*]] = sdy.reshard %arg2 <@mesh_xyztp, [{"y":(2)2}, {}, {"x":(2)2}, {}]> : tensor
  // CHECK-DAG: %[[SCATTER:.*]] = "stablehlo.scatter"(%arg0, %[[RESHARD1]], %[[RESHARD2]])
  // CHECK-DAG: return %[[SCATTER]] : tensor
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 3],
      inserted_window_dims = [2],
      input_batching_dims = [1, 3],
      scatter_indices_batching_dims = [1, 0],
      scatter_dims_to_operand_dims = [2, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyztp, [{"x": (1)2}, {"x": (2)2}, {"y": (1)2}, {"y": (2)2}, {"z": (1)2}]>]>
  } : (tensor<6x4x10x12x14xf32>, tensor<12x4x2xi64>, tensor<12x2x4x10xf32>) -> tensor<6x4x10x12x14xf32>
  return %0 : tensor<6x4x10x12x14xf32>
}
