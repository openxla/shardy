// RUN: sdy_opt %s -sdy-pad-for-divisibility -split-input-file | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Masks padded i32 scatter_indices slots with -1 so high side zero padding does
// not scatter phantom updates into index 0.
// CHECK-LABEL: func @scatter_padded_indices_i32
func.func @scatter_padded_indices_i32(
    %arg0: tensor<6x4xf32>, %arg1: tensor<5x1xi32>, %arg2: tensor<5x4xf32>)
    -> tensor<6x4xf32> {
  // CHECK-DAG:  %[[ZERO_I32:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG:  %[[PAD_IDX:.*]] = stablehlo.pad %arg1, %[[ZERO_I32]], low = [0, 0], high = [3, 0], interior = [0, 0] : (tensor<5x1xi32>, tensor<i32>) -> tensor<8x1xi32>
  // CHECK-DAG:  %[[SLICE_IDX:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD_IDX]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x1xi32>
  // CHECK-DAG:  %[[ZERO_F32:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG:  %[[PAD_UPD:.*]] = stablehlo.pad %arg2, %[[ZERO_F32]], low = [0, 0], high = [3, 0], interior = [0, 0] : (tensor<5x4xf32>, tensor<f32>) -> tensor<8x4xf32>
  // CHECK-DAG:  %[[SLICE_UPD:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD_UPD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x4xf32>
  // CHECK-DAG:  %[[NEG_ONE:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // CHECK-DAG:  %[[IOTA:.*]] = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<8x1xi32>
  // CHECK-DAG:  %[[LIMIT:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK-DAG:  %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<i32>) -> tensor<8x1xi32>
  // CHECK-DAG:  %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<8x1xi32>, tensor<8x1xi32>) -> tensor<8x1xi1>
  // CHECK-DAG:  %[[BCAST_OOB:.*]] = stablehlo.broadcast_in_dim %[[NEG_ONE]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<i32>) -> tensor<8x1xi32>
  // CHECK-DAG:  %[[MASKED_IDX:.*]] = stablehlo.select %[[MASK]], %[[SLICE_IDX]], %[[BCAST_OOB]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<8x1xi1>, tensor<8x1xi32>
  // CHECK:      %[[SCATTER:.*]] = "stablehlo.scatter"(%arg0, %[[MASKED_IDX]], %[[SLICE_UPD]])
  // CHECK:      : (tensor<6x4xf32>, tensor<8x1xi32>, tensor<8x4xf32>) -> tensor<6x4xf32>
  // CHECK:      return %[[SCATTER]] : tensor<6x4xf32>
  %idx = sdy.all_slice [{"x"}, {}] %arg1 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<5x1xi32>
  %upd = sdy.all_slice [{"x"}, {}] %arg2 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<5x4xf32>
  %0 = "stablehlo.scatter"(%arg0, %idx, %upd) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<6x4xf32>, tensor<5x1xi32>, tensor<5x4xf32>) -> tensor<6x4xf32>
  return %0 : tensor<6x4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Masks padded ui32 scatter_indices slots with UINT32_MAX (4294967295).
// CHECK-LABEL: func @scatter_padded_indices_ui32
func.func @scatter_padded_indices_ui32(
    %arg0: tensor<6x4xf32>, %arg1: tensor<5x1xui32>, %arg2: tensor<5x4xf32>)
    -> tensor<6x4xf32> {
  // CHECK-DAG:  %[[UINT_MAX:.*]] = stablehlo.constant dense<4294967295> : tensor<ui32>
  // CHECK-DAG:  %[[BCAST_OOB:.*]] = stablehlo.broadcast_in_dim %[[UINT_MAX]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<ui32>) -> tensor<8x1xui32>
  // CHECK:      %[[MASKED_IDX:.*]] = stablehlo.select %{{.*}}, %{{.*}}, %[[BCAST_OOB]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<8x1xi1>, tensor<8x1xui32>
  // CHECK:      %[[SCATTER:.*]] = "stablehlo.scatter"(%arg0, %[[MASKED_IDX]], %{{.*}})
  // CHECK:      : (tensor<6x4xf32>, tensor<8x1xui32>, tensor<8x4xf32>) -> tensor<6x4xf32>
  // CHECK:      return %[[SCATTER]] : tensor<6x4xf32>
  %idx = sdy.all_slice [{"x"}, {}] %arg1 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<5x1xui32>
  %upd = sdy.all_slice [{"x"}, {}] %arg2 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<5x4xf32>
  %0 = "stablehlo.scatter"(%arg0, %idx, %upd) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %min = stablehlo.minimum %lhs, %rhs : tensor<f32>
    stablehlo.return %min : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}]>]>
  } : (tensor<6x4xf32>, tensor<5x1xui32>, tensor<5x4xf32>) -> tensor<6x4xf32>
  return %0 : tensor<6x4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Handles variadic scatter where scatter_indices is at operand position
// op.getInputs().size() == 2.
// CHECK-LABEL: func @scatter_variadic_padded_indices_2d_coords
func.func @scatter_variadic_padded_indices_2d_coords(
    %in0: tensor<4x4xi32>, %in1: tensor<4x4xf32>,
    %arg_idx: tensor<3x2xi64>,
    %u0: tensor<3xi32>, %u1: tensor<3xf32>)
    -> (tensor<4x4xi32>, tensor<4x4xf32>) {
  // CHECK-DAG:  %[[NEG_ONE:.*]] = stablehlo.constant dense<-1> : tensor<i64>
  // CHECK-DAG:  %[[BCAST_OOB:.*]] = stablehlo.broadcast_in_dim %[[NEG_ONE]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {}]>]>} : (tensor<i64>) -> tensor<4x2xi64>
  // CHECK:      %[[MASKED_IDX:.*]] = stablehlo.select %{{.*}}, %{{.*}}, %[[BCAST_OOB]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {}]>]>} : tensor<4x2xi1>, tensor<4x2xi64>
  // CHECK:      %[[SCATTER:.*]]:2 = "stablehlo.scatter"(%arg0, %arg1, %[[MASKED_IDX]], %{{.*}}, %{{.*}})
  // CHECK:      : (tensor<4x4xi32>, tensor<4x4xf32>, tensor<4x2xi64>, tensor<4xi32>, tensor<4xf32>) -> (tensor<4x4xi32>, tensor<4x4xf32>)
  // CHECK:      return %[[SCATTER]]#0, %[[SCATTER]]#1
  %idx = sdy.all_slice [{"y"}, {}] %arg_idx out_sharding=<@mesh_4_2, [{"y"}, {}]> : tensor<3x2xi64>
  %upd0 = sdy.all_slice [{"y"}] %u0 out_sharding=<@mesh_4_2, [{"y"}]> : tensor<3xi32>
  %upd1 = sdy.all_slice [{"y"}] %u1 out_sharding=<@mesh_4_2, [{"y"}]> : tensor<3xf32>
  %0:2 = "stablehlo.scatter"(%in0, %in1, %idx, %upd0, %upd1) ({
  ^bb0(%l0: tensor<i32>, %l1: tensor<f32>, %r0: tensor<i32>, %r1: tensor<f32>):
    %s0 = stablehlo.add %l0, %r0 : tensor<i32>
    %s1 = stablehlo.add %l1, %r1 : tensor<f32>
    stablehlo.return %s0, %s1 : tensor<i32>, tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {}], unreduced={"y"}>, <@mesh_4_2, [{}, {}], unreduced={"y"}>]>
  } : (tensor<4x4xi32>, tensor<4x4xf32>, tensor<3x2xi64>, tensor<3xi32>, tensor<3xf32>) -> (tensor<4x4xi32>, tensor<4x4xf32>)
  return %0#0, %0#1 : tensor<4x4xi32>, tensor<4x4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Does not emit index masking when scatter_indices is already divisible.
// CHECK-LABEL: func @scatter_divisible_indices_no_mask
func.func @scatter_divisible_indices_no_mask(
    %arg0: tensor<5x4xf32>, %arg1: tensor<4x1xi32>, %arg2: tensor<4x4xf32>)
    -> tensor<5x4xf32> {
  // CHECK-NOT:  stablehlo.select
  // CHECK:      "stablehlo.scatter"
  %in = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<5x4xf32>
  %idx = sdy.all_slice [{"x"}, {}] %arg1 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<4x1xi32>
  %upd = sdy.all_slice [{"x"}, {}] %arg2 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<4x4xf32>
  %0 = "stablehlo.scatter"(%in, %idx, %upd) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>
  } : (tensor<5x4xf32>, tensor<4x1xi32>, tensor<4x4xf32>) -> tensor<5x4xf32>
  %ag = sdy.all_gather [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<5x4xf32>
  return %ag : tensor<5x4xf32>
}
