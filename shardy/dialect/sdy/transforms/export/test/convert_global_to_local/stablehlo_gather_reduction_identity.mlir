// RUN: sdy_opt %s --sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh_2 = <["x"=2]>

// CHECK-LABEL: func @gather_unreduced_max_i1_uses_false_identity(
func.func @gather_unreduced_max_i1_uses_false_identity(
  %arg0: tensor<8x4xi1> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2xi64>)
  -> (tensor<2x4xi1> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}], unreduced=max{"x"}>}) {
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%arg0, %{{.*}})
  // CHECK: %[[ID:.*]] = stablehlo.constant dense<false> : tensor<2x4xi1>
  // CHECK: %[[SEL:.*]] = stablehlo.select %{{.*}}, %[[GATHER]], %[[ID]]
  // CHECK: return %[[SEL]] : tensor<2x4xi1>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced=max{"x"}>]>
  } : (tensor<8x4xi1>, tensor<2xi64>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

// CHECK-LABEL: func @gather_unreduced_min_i1_uses_true_identity(
func.func @gather_unreduced_min_i1_uses_true_identity(
  %arg0: tensor<8x4xi1> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2xi64>)
  -> (tensor<2x4xi1> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}], unreduced=min{"x"}>}) {
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%arg0, %{{.*}})
  // CHECK: %[[ID:.*]] = stablehlo.constant dense<true> : tensor<2x4xi1>
  // CHECK: %[[SEL:.*]] = stablehlo.select %{{.*}}, %[[GATHER]], %[[ID]]
  // CHECK: return %[[SEL]] : tensor<2x4xi1>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced=min{"x"}>]>
  } : (tensor<8x4xi1>, tensor<2xi64>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

// CHECK-LABEL: func @gather_unreduced_max_f8e4m3fn_uses_min_finite_identity(
func.func @gather_unreduced_max_f8e4m3fn_uses_min_finite_identity(
  %arg0: tensor<8x4xf8E4M3FN> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2xi64>)
  -> (tensor<2x4xf8E4M3FN> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}], unreduced=max{"x"}>}) {
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%arg0, %{{.*}})
  // CHECK: %[[ID:.*]] = stablehlo.constant dense<-4.480000e+02> : tensor<2x4xf8E4M3FN>
  // CHECK: %[[SEL:.*]] = stablehlo.select %{{.*}}, %[[GATHER]], %[[ID]]
  // CHECK: return %[[SEL]] : tensor<2x4xf8E4M3FN>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced=max{"x"}>]>
  } : (tensor<8x4xf8E4M3FN>, tensor<2xi64>) -> tensor<2x4xf8E4M3FN>
  return %0 : tensor<2x4xf8E4M3FN>
}

// CHECK-LABEL: func @gather_unreduced_max_f8e8m0fnu_uses_smallest_identity(
func.func @gather_unreduced_max_f8e8m0fnu_uses_smallest_identity(
  %arg0: tensor<8x4xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2xi64>)
  -> (tensor<2x4xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}], unreduced=max{"x"}>}) {
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%arg0, %{{.*}})
  // CHECK: %[[ID:.*]] = stablehlo.constant dense<5.877470e-39> : tensor<2x4xf8E8M0FNU>
  // CHECK: %[[SEL:.*]] = stablehlo.select %{{.*}}, %[[GATHER]], %[[ID]]
  // CHECK: return %[[SEL]] : tensor<2x4xf8E8M0FNU>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced=max{"x"}>]>
  } : (tensor<8x4xf8E8M0FNU>, tensor<2xi64>) -> tensor<2x4xf8E8M0FNU>
  return %0 : tensor<2x4xf8E8M0FNU>
}
