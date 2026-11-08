// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_6 = <["x"=6]>
sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// CHECK-LABEL: func @padded_gather_i8_201_widens_to_i32(
func.func @padded_gather_i8_201_widens_to_i32(
    %arg0: tensor<201x4xf32>, %arg1: tensor<3x1xi8>) -> tensor<3x4xf32> {
  // CHECK: %[[IDX_I32:.*]] = stablehlo.convert %arg1 : (tensor<3x1xi8>) -> tensor<3x1xi32>
  // CHECK: %[[CST_MAX:.*]] = stablehlo.constant dense<200> : tensor<1xi32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CST_MAX]], dims = [1] : (tensor<1xi32>) -> tensor<3x1xi32>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.minimum %[[IDX_I32]], %[[BCAST]] : tensor<3x1xi32>
  // CHECK: "stablehlo.gather"(%{{.*}}, %[[CLAMPED]])
  // CHECK-SAME: : (tensor<202x4xf32>, tensor<3x1xi32>) -> tensor<3x4xf32>
  %sliced = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_2, [{"x"}, {}]> : tensor<201x4xf32>
  %0 = "stablehlo.gather"(%sliced, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<201x4xf32>, tensor<3x1xi8>) -> tensor<3x4xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// CHECK-LABEL: func @padded_gather_ui16_70001_implicit_ivd_widens_to_i32(
func.func @padded_gather_ui16_70001_implicit_ivd_widens_to_i32(
    %arg0: tensor<70001x4xf32>,
    %arg1: tensor<2xui16> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}]>})
    -> tensor<2x4xf32> {
  // CHECK: %[[IDX_I32:.*]] = stablehlo.convert %{{.*}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}]>]>} : (tensor<2xui16>) -> tensor<2xi32>
  // CHECK: %[[CST_MAX:.*]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}]>]>} dense<70000> : tensor<2xi32>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.minimum %[[IDX_I32]], %[[CST_MAX]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}]>]>} : tensor<2xi32>
  // CHECK: "stablehlo.gather"(%{{.*}}, %[[CLAMPED]])
  // CHECK-SAME: : (tensor<70002x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
  %sliced = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_2_2, [{"x"}, {}]> : tensor<70001x4xf32>
  %0 = "stablehlo.gather"(%sliced, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}], unreduced={"x"}>]>
  } : (tensor<70001x4xf32>, tensor<2xui16>) -> tensor<2x4xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_2, [{"y"}, {}]> : tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// CHECK-LABEL: func @padded_gather_ui8_255_padded_to_258_widens_to_i32(
func.func @padded_gather_ui8_255_padded_to_258_widens_to_i32(
    %arg0: tensor<255x4xf32>, %arg1: tensor<2x1xui8>) -> tensor<2x4xf32> {
  // Even though unpadded maxIdx (254) fits in ui8, padded dim - 1 (257) exceeds
  // ui8 and must widen to i32 so ConvertGlobalToLocal does not truncate 257.
  // CHECK: %[[IDX_I32:.*]] = stablehlo.convert %arg1 : (tensor<2x1xui8>) -> tensor<2x1xi32>
  // CHECK: %[[CST_MAX:.*]] = stablehlo.constant dense<254> : tensor<1xi32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CST_MAX]], dims = [1] : (tensor<1xi32>) -> tensor<2x1xi32>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.minimum %[[IDX_I32]], %[[BCAST]] : tensor<2x1xi32>
  // CHECK: "stablehlo.gather"(%{{.*}}, %[[CLAMPED]])
  // CHECK-SAME: : (tensor<258x4xf32>, tensor<2x1xi32>) -> tensor<2x4xf32>
  %sliced = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_6, [{"x"}, {}]> : tensor<255x4xf32>
  %0 = "stablehlo.gather"(%sliced, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{}, {}], unreduced={"x"}>]>
  } : (tensor<255x4xf32>, tensor<2x1xui8>) -> tensor<2x4xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_6, [{}, {}]> : tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// CHECK-LABEL: func @padded_gather_i32_large_dim_widens_to_i64(
func.func @padded_gather_i32_large_dim_widens_to_i64(
    %arg0: tensor<2147483649x1xf32>, %arg1: tensor<2x1xi32>) -> tensor<2x1xf32> {
  // CHECK: %[[IDX_I64:.*]] = stablehlo.convert %arg1 : (tensor<2x1xi32>) -> tensor<2x1xi64>
  // CHECK: %[[CST_MAX:.*]] = stablehlo.constant dense<2147483648> : tensor<1xi64>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CST_MAX]], dims = [1] : (tensor<1xi64>) -> tensor<2x1xi64>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.minimum %[[IDX_I64]], %[[BCAST]] : tensor<2x1xi64>
  // CHECK: "stablehlo.gather"(%{{.*}}, %[[CLAMPED]])
  %sliced = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_2, [{"x"}, {}]> : tensor<2147483649x1xf32>
  %0 = "stablehlo.gather"(%sliced, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<2147483649x1xf32>, tensor<2x1xi32>) -> tensor<2x1xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}

// CHECK-LABEL: func @padded_gather_i8_within_bounds_stays_i8(
func.func @padded_gather_i8_within_bounds_stays_i8(
    %arg0: tensor<101x4xf32>, %arg1: tensor<2x1xi8>) -> tensor<2x4xf32> {
  // CHECK-NOT: stablehlo.convert
  // CHECK: %[[CST_MAX:.*]] = stablehlo.constant dense<100> : tensor<1xi8>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CST_MAX]], dims = [1] : (tensor<1xi8>) -> tensor<2x1xi8>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.minimum %arg1, %[[BCAST]] : tensor<2x1xi8>
  // CHECK: "stablehlo.gather"(%{{.*}}, %[[CLAMPED]])
  %sliced = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_2, [{"x"}, {}]> : tensor<101x4xf32>
  %0 = "stablehlo.gather"(%sliced, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<101x4xf32>, tensor<2x1xi8>) -> tensor<2x4xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// CHECK-LABEL: func @padded_gather_while_block_arg_indices_preserves_sharding(
func.func @padded_gather_while_block_arg_indices_preserves_sharding(
    %arg0: tensor<201x4xf32>,
    %arg1: tensor<4x1xi8> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {}]>})
    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"y"}, {}]>}) {
  // CHECK: stablehlo.while
  // CHECK: do {
  // CHECK: %[[IDX_I32:.*]] = stablehlo.convert %{{.*}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>} : (tensor<4x1xi8>) -> tensor<4x1xi32>
  // CHECK: %[[CST_MAX:.*]] = stablehlo.constant dense<200> : tensor<1xi32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CST_MAX]], dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>} : (tensor<1xi32>) -> tensor<4x1xi32>
  // CHECK: %[[CLAMPED:.*]] = stablehlo.minimum %[[IDX_I32]], %[[BCAST]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>} : tensor<4x1xi32>
  // CHECK: "stablehlo.gather"(%{{.*}}, %[[CLAMPED]])
  %sliced = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_2_2, [{"x"}, {}]> : tensor<201x4xf32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %init_acc = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>} dense<0.0> : tensor<4x4xf32>
  %w:4 = stablehlo.while(%iter_tbl = %sliced, %iter_idx = %arg1, %iter_acc = %init_acc, %iter_i = %c0)
      : tensor<201x4xf32>, tensor<4x1xi8>, tensor<4x4xf32>, tensor<i32>
    attributes {sdy.sharding = #sdy.sharding_per_value<[
      <@mesh_2_2, [{"x"}, {}]>, <@mesh_2_2, [{"y"}, {}]>, <@mesh_2_2, [{"y"}, {}]>, <@mesh_2_2, []>]>}
    cond {
      %cond = stablehlo.compare LT, %iter_i, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %g = "stablehlo.gather"(%iter_tbl, %iter_idx) {
        dimension_numbers = #stablehlo.gather<
          offset_dims = [1], collapsed_slice_dims = [0],
          start_index_map = [0], index_vector_dim = 1>,
        slice_sizes = array<i64: 1, 4>,
        sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}], unreduced={"x"}>]>
      } : (tensor<201x4xf32>, tensor<4x1xi8>) -> tensor<4x4xf32>
      %r = sdy.all_reduce {"x"} %g out_sharding=<@mesh_2_2, [{"y"}, {}]> : tensor<4x4xf32>
      %next_acc = stablehlo.add %iter_acc, %r
        {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"y"}, {}]>]>} : tensor<4x4xf32>
      %next_i = stablehlo.add %iter_i, %c1 : tensor<i32>
      stablehlo.return %iter_tbl, %iter_idx, %next_acc, %next_i
        : tensor<201x4xf32>, tensor<4x1xi8>, tensor<4x4xf32>, tensor<i32>
    }
  return %w#2 : tensor<4x4xf32>
}
