// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>
sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=8]>

// CHECK-LABEL: func @sort
// TODO(b/479473118): Shardings with fully replication should not be open.
func.func @sort(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x32x8xf32>) -> (tensor<4x32x8xi32>, tensor<4x32x8xf32>) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}, {}]>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}, {}]>
  // CHECK-NEXT: %[[SORT:.*]]:2 = "stablehlo.sort"(%[[RESHARD0]], %[[RESHARD1]])
  // CHECK: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}]>, <@mesh, [{}, {"x"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %2#0 <@mesh, [{}, {}, {}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %2#1 <@mesh, [{?}, {?}, {?}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = "stablehlo.sort"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<4x32x8xi32>, tensor<4x32x8xf32>) -> (tensor<4x32x8xi32>, tensor<4x32x8xf32>)
  return %0#0, %0#1 : tensor<4x32x8xi32>, tensor<4x32x8xf32>
}

// CHECK-LABEL: func @sort_all_other_dims_size_one
func.func @sort_all_other_dims_size_one(%arg0: tensor<1x4x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<1x4x1xi32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}]> : tensor<1x4x1xi32>
  // CHECK-NEXT: "stablehlo.sort"(%[[RESHARD]])
  %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 1 : i64, is_stable = true} : (tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
  return %0 : tensor<1x4x1xi32>
}

// CHECK-LABEL: func @sort_single_input_output
func.func @sort_single_input_output(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<4x32x8xi32>) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y", "x"}, {}]>
  // CHECK-NEXT: %[[SORT:.*]] = "stablehlo.sort"(%[[RESHARD0]])
  // CHECK: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[SORT]] <@mesh, [{}, {}, {}]>
  // CHECK=NEXT: return %[[RESHARD1]]
  %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_compatible
func.func @sort_compatible(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_input_and_output_shardings_are_same_on_sorting_dimension
func.func @sort_input_and_output_shardings_are_same_on_sorting_dimension(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y", "x"}, {}]> : tensor<4x32x8xi32>
  // CHECK-NEXT: %[[SORT:.*]] = "stablehlo.sort"(%[[RESHARD1]])
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %[[SORT]] <@mesh, [{"x"}, {"y"}, {}]> : tensor<4x32x8xi32>
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_input_and_output_shardings_are_different_on_sorting_dimension
func.func @sort_input_and_output_shardings_are_different_on_sorting_dimension(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {"z"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {"z", "x"}, {}]>
  // CHECK-NEXT: %[[SORT:.*]] = "stablehlo.sort"(%[[RESHARD0]])
  // CHECK: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"z", "x"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[SORT]] <@mesh_xyz, [{"y"}, {"z"}, {}]>
  // CHECK-NEXT: return %[[RESHARD1]]
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {"z"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_sorting_dim_shardings_has_common_prefix
func.func @sort_sorting_dim_shardings_has_common_prefix(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y", "z"}, {"x"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y", "t"}, {"z"}, {}]>}) {
  // CHECK: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {"z", "y"}, {"t"}]>
  // CHECK-NEXT: %[[SORT:.*]] = "stablehlo.sort"(%[[RESHARD0]])
  // CHECK: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"z", "y"}, {"t"}]>]>}
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[SORT]] <@mesh_xyzt, [{"y", "t"}, {"z"}, {}]>
  // CHECK-NEXT: return %[[RESHARD1]]
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y", "t"}, {"z"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_sorting_dim_shardings_has_common_prefix_and_large
func.func @sort_sorting_dim_shardings_has_common_prefix_and_large(%arg0: tensor<64x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y", "t", "z"}, {"x"}, {}]>}) -> (tensor<64x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y", "t"}, {"z"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {"z", "y"}, {"t"}]> : tensor<64x32x8xi32>
  // CHECK-NEXT: %[[SORT:.*]] = "stablehlo.sort"(%[[RESHARD1]])
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %[[SORT]] <@mesh_xyzt, [{"y", "t"}, {"z"}, {}]> : tensor<64x32x8xi32>
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y", "t"}, {"z"}, {}]>]>} : (tensor<64x32x8xi32>) -> (tensor<64x32x8xi32>)
  return %0 : tensor<64x32x8xi32>
}

// CHECK-LABEL: func @sort_incompatible_on_nonsort_dimensions
func.func @sort_incompatible_on_nonsort_dimensions(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) {
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %0 <@mesh, [{}, {"y"}, {}]> : tensor<4x32x8xi32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x32x8xi32>
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_compatible_on_nonsort_dimension
func.func @sort_compatible_on_nonsort_dimension(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}
