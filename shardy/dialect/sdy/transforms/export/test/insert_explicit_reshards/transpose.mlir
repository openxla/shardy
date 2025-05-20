// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=3]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>

// CHECK-LABEL: func @transpose_simple_compatible
func.func @transpose_simple_compatible(%arg0: tensor<8x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<12x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.transpose %arg0, dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  return %0 : tensor<12x8xf32>
}

// CHECK-LABEL: func @transpose_simple_incompatible
func.func @transpose_simple_incompatible(%arg0: tensor<8x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<12x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK: %[[TRANSPOSE:.*]] = stablehlo.transpose %arg0, dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[TRANSPOSE]] <@mesh, [{"x"}, {"y"}]> : tensor<12x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<12x8xf32>
  %0 = stablehlo.transpose %arg0, dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  return %0 : tensor<12x8xf32>
}

// CHECK-LABEL: func @transpose_tensor_with_four_dimensions
func.func @transpose_tensor_with_four_dimensions(%arg0: tensor<256x32x64x100xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x"}, {"y"}, {}]>}) -> (tensor<100x32x256x64xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"y"}, {"z"}, {}]>}) {
  // CHECK: %[[TRANSPOSE:.*]] = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x"}, {}, {"y"}]>]>} : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[TRANSPOSE]] <@mesh_xyz, [{}, {"y"}, {"z"}, {}]> : tensor<100x32x256x64xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<100x32x256x64xf32>
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"y"}, {"z"}, {}]>]>} : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
  return %0 : tensor<100x32x256x64xf32>
}

// CHECK-LABEL: func @transpose_after_dot_used_by_transpose_and_negate
func.func @transpose_after_dot_used_by_transpose_and_negate(%arg0: tensor<8x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<12x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x12xf32>, tensor<12x12xf32>) -> tensor<8x12xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x12xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x"}, {"y"}]> : tensor<8x12xf32>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[RESHARD]], dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x12xf32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x12xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x12xf32>, tensor<12x12xf32>) -> tensor<8x12xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  %2 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x12xf32>
  return %2 : tensor<8x12xf32>
}

// CHECK-LABEL: func @transpose_after_dot_only_use_is_transpose
func.func @transpose_after_dot_only_use_is_transpose(%arg0: tensor<8x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<12x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<12x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x12xf32>, tensor<12x12xf32>) -> tensor<8x12xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x12xf32>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ALL_REDUCE]], dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[TRANSPOSE]] <@mesh, [{"y"}, {"x"}]> : tensor<12x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<12x8xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x12xf32>, tensor<12x12xf32>) -> tensor<8x12xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  return %1 : tensor<12x8xf32>
}

// CHECK-LABEL: func @transpose_after_dot_used_by_multiple_transpose_and_only_use_is_transpose
func.func @transpose_after_dot_used_by_multiple_transpose_and_only_use_is_transpose(%arg0: tensor<8x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<12x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<12x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x12xf32>, tensor<12x12xf32>) -> tensor<8x12xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x12xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x"}, {"y"}]> : tensor<8x12xf32>
  // CHECK-NEXT: %[[TRANSPOSE1:.*]] = stablehlo.transpose %[[RESHARD]], dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  // CHECK-NEXT: %[[TRANSPOSE2:.*]] = stablehlo.transpose %[[RESHARD]], dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  // CHECK-NEXT: return %[[TRANSPOSE2]] : tensor<12x8xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x12xf32>, tensor<12x12xf32>) -> tensor<8x12xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  %2 = stablehlo.transpose %0, dims = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x12xf32>) -> tensor<12x8xf32>
  return %2 : tensor<12x8xf32>
}
