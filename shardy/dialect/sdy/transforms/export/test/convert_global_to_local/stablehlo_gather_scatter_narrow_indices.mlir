// RUN: sdy_opt %s --sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh_2 = <["x"=2]>

// CHECK-LABEL: func @gather_i8_200_widens_clamp_and_bounds_to_i32(
func.func @gather_i8_200_widens_clamp_and_bounds_to_i32(
    %arg0: tensor<200x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<3xi8>) -> tensor<3x4xf32> {
  // CHECK: %[[IDX_I32:.*]] = stablehlo.convert %arg1 : (tensor<3xi8>) -> tensor<3xi32>
  // CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<3xi32>
  // CHECK-DAG: %[[MAX:.*]] = stablehlo.constant dense<199> : tensor<3xi32>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.clamp %[[ZERO]], %[[IDX_I32]], %[[MAX]] : tensor<3xi32>
  // CHECK: %[[SHARD_MAX_OFF:.*]] = stablehlo.constant dense<99> : tensor<i32>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %{{.*}} : tensor<3xi32>
  // CHECK: "stablehlo.gather"(%arg0, %[[LOCAL_IDX]])
  // CHECK-SAME: (tensor<100x4xf32>, tensor<3xi32>) -> tensor<3x4xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<200x4xf32>, tensor<3xi8>) -> tensor<3x4xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// CHECK-LABEL: func @gather_ui32_saturates_before_i32_cast(
func.func @gather_ui32_saturates_before_i32_cast(
    %arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<2xui32>) -> tensor<2x4xf32> {
  // CHECK: %[[CAP:.*]] = stablehlo.constant dense<2147483647> : tensor<2xui32>
  // CHECK: %[[SAT:.*]] = stablehlo.minimum %arg1, %[[CAP]] : tensor<2xui32>
  // CHECK: %[[IDX_I32:.*]] = stablehlo.convert %[[SAT]] : (tensor<2xui32>) -> tensor<2xi32>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.clamp %{{.*}}, %[[IDX_I32]], %{{.*}} : tensor<2xi32>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %{{.*}} : tensor<2xi32>
  // CHECK: "stablehlo.gather"(%arg0, %[[LOCAL_IDX]])
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<16x4xf32>, tensor<2xui32>) -> tensor<2x4xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// CHECK-LABEL: func @gather_large_global_dim_widens_i32_to_i64(
func.func @gather_large_global_dim_widens_i32_to_i64(
    %arg0: tensor<2147483648x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<2xi32>) -> tensor<2x1xf32> {
  // CHECK: %[[IDX_I64:.*]] = stablehlo.convert %arg1 : (tensor<2xi32>) -> tensor<2xi64>
  // CHECK: %[[MAX:.*]] = stablehlo.constant dense<2147483647> : tensor<2xi64>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.clamp %{{.*}}, %[[IDX_I64]], %[[MAX]] : tensor<2xi64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %{{.*}} : tensor<2xi64>
  // CHECK: "stablehlo.gather"(%arg0, %[[LOCAL_IDX]])
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<2147483648x1xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}

// CHECK-LABEL: func @scatter_ui16_128k_widens_indices_to_i32(
func.func @scatter_ui16_128k_widens_indices_to_i32(
    %arg0: tensor<131072x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<2x1xui16>,
    %arg2: tensor<2x4xf32>)
    -> (tensor<131072x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  // CHECK: %[[IDX_I32:.*]] = stablehlo.convert %arg1 : (tensor<2x1xui16>) -> tensor<2x1xi32>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[IDX_I32]], %{{.*}} : tensor<2x1xi32>
  // CHECK: "stablehlo.scatter"(%arg0, %[[LOCAL_IDX]], %arg2)
  // CHECK: (tensor<65536x4xf32>, tensor<2x1xi32>, tensor<2x4xf32>) -> tensor<65536x4xf32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<131072x4xf32>, tensor<2x1xui16>, tensor<2x4xf32>) -> tensor<131072x4xf32>
  return %0 : tensor<131072x4xf32>
}

// CHECK-LABEL: func @scatter_ui64_saturates_before_i64_cast(
func.func @scatter_ui64_saturates_before_i64_cast(
    %arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<2x1xui64>,
    %arg2: tensor<2x4xf32>)
    -> (tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  // CHECK: %[[CAP:.*]] = stablehlo.constant dense<9223372036854775807> : tensor<2x1xui64>
  // CHECK: %[[SAT:.*]] = stablehlo.minimum %arg1, %[[CAP]] : tensor<2x1xui64>
  // CHECK: %[[IDX_I64:.*]] = stablehlo.convert %[[SAT]] : (tensor<2x1xui64>) -> tensor<2x1xi64>
  // CHECK: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[IDX_I64]], %{{.*}} : tensor<2x1xi64>
  // CHECK: "stablehlo.scatter"(%arg0, %[[LOCAL_IDX]], %arg2)
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<16x4xf32>, tensor<2x1xui64>, tensor<2x4xf32>) -> tensor<16x4xf32>
  return %0 : tensor<16x4xf32>
}
