// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning="filter=dot,constant,reshard,all_gather,all_slice" | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @selective_dot
// CHECK-SAME: (%[[LHS:.*]]: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %[[RHS:.*]]: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
func.func @selective_dot(%lhs: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
                         %rhs: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[LHS]], %[[RHS]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {"y"}]>]
  // CHECK-SAME:   manual_axes={"x", "y"} (%arg2: tensor<4x32xf32>, %arg3: tensor<32x8xf32>) {
  // CHECK-NEXT:   %[[LOCAL_DOT:.*]] = stablehlo.dot %arg2, %arg3 : (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT:   sdy.return %[[LOCAL_DOT]] : tensor<4x8xf32>
  // CHECK-NEXT: } : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %dot = stablehlo.dot %lhs, %rhs {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>

  // CHECK: %[[ADD:.*]] = stablehlo.add %[[MANUAL]], %[[MANUAL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x16xf32>
  %add = stablehlo.add %dot, %dot {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x16xf32>

  // CHECK: return %[[ADD]] : tensor<8x16xf32>
  return %add : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @selective_indivisible_dot
// CHECK-SAME: (%[[LHS:.*]]: tensor<6x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %[[RHS:.*]]: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
func.func @selective_indivisible_dot(%lhs: tensor<6x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
                                     %rhs: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> (tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[LHS]] [0:5, 0:32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<6x32xf32>) -> tensor<5x32xf32>
  %sliced_lhs = stablehlo.slice %lhs [0:5, 0:32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<6x32xf32>) -> tensor<5x32xf32>

  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:      %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<5x32xf32>, tensor<f32>) -> tensor<6x32xf32>
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[PAD]], %[[RHS]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg2: tensor<3x32xf32>, %arg3: tensor<32x16xf32>) {
  // CHECK-NEXT:   %[[LOCAL_DOT:.*]] = stablehlo.dot %arg2, %arg3 : (tensor<3x32xf32>, tensor<32x16xf32>) -> tensor<3x16xf32>
  // CHECK-NEXT:   sdy.return %[[LOCAL_DOT]] : tensor<3x16xf32>
  // CHECK-NEXT: } : (tensor<6x32xf32>, tensor<32x16xf32>) -> tensor<6x16xf32>
  // CHECK-NEXT: %[[SLICED_RES:.*]] = stablehlo.slice %[[MANUAL]] [0:5, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<6x16xf32>) -> tensor<5x16xf32>
  %dot = stablehlo.dot %sliced_lhs, %rhs {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<5x32xf32>, tensor<32x16xf32>) -> tensor<5x16xf32>

  // CHECK:      %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:      %[[PAD_RESHARD:.*]] = stablehlo.pad %[[SLICED_RES]], %[[CST_0]], low = [0, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<5x16xf32>, tensor<f32>) -> tensor<6x16xf32>
  // CHECK:      %[[MANUAL_RESHARD:.*]] = sdy.manual_computation(%[[PAD_RESHARD]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg2: tensor<3x16xf32>) {
  // CHECK-NEXT:   %[[ALL_GATHER:.*]] = "stablehlo.all_gather"(%arg2)
  // CHECK-SAME:     replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = [#stablehlo.axis_ref<name = "x">]>
  // CHECK-NEXT:   sdy.return %[[ALL_GATHER]] : tensor<6x16xf32>
  // CHECK-NEXT: } : (tensor<6x16xf32>) -> tensor<6x16xf32>
  // CHECK-NEXT: %[[SLICED_FINAL:.*]] = stablehlo.slice %[[MANUAL_RESHARD]] [0:5, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x16xf32>) -> tensor<5x16xf32>
  %res = sdy.reshard %dot <@mesh, [{}, {}]> : tensor<5x16xf32>

  // CHECK: return %[[SLICED_FINAL]] : tensor<5x16xf32>
  return %res : tensor<5x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @selective_sharded_constant
// CHECK-SAME: () -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @selective_sharded_constant()
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation()
  // CHECK-SAME:   in_shardings=[]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} () {
  // CHECK-NEXT:   %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4x8xf32>
  // CHECK-NEXT:   sdy.return %[[CST]] : tensor<4x8xf32>
  // CHECK-NEXT: } : () -> tensor<8x8xf32>
  %c = "sdy.constant"() <{value = dense<1.0> : tensor<8x8xf32>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : () -> tensor<8x8xf32>
  // CHECK: return %[[MANUAL]] : tensor<8x8xf32>
  return %c : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @selective_indivisible_reshard
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<5x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
func.func @selective_indivisible_reshard(%arg0: tensor<5x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<5x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:      %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<5x8xf32>, tensor<f32>) -> tensor<6x8xf32>
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[PAD]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg1: tensor<3x8xf32>) {
  // CHECK-NEXT:   %[[ALL_GATHER:.*]] = "stablehlo.all_gather"(%arg1)
  // CHECK-SAME:     replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = [#stablehlo.axis_ref<name = "x">]>
  // CHECK-NEXT:   sdy.return %[[ALL_GATHER]] : tensor<6x8xf32>
  // CHECK-NEXT: } : (tensor<6x8xf32>) -> tensor<6x8xf32>
  // CHECK-NEXT: %[[SLICED:.*]] = stablehlo.slice %[[MANUAL]] [0:5, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x8xf32>) -> tensor<5x8xf32>
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<5x8xf32>
  // CHECK: return %[[SLICED]] : tensor<5x8xf32>
  return %0 : tensor<5x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @selective_all_gather
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
func.func @selective_all_gather(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[ARG0]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg1: tensor<4x16xf32>) {
  // CHECK-NEXT:   %[[AG:.*]] = "stablehlo.all_gather"(%arg1)
  // CHECK-SAME:     all_gather_dim = 0 : i64
  // CHECK-SAME:     replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = [#stablehlo.axis_ref<name = "x">]>
  // CHECK-NEXT:   sdy.return %[[AG]] : tensor<8x16xf32>
  // CHECK-NEXT: } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK: return %[[MANUAL]] : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @selective_all_slice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @selective_all_slice(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[ARG0]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg1: tensor<8x16xf32>) {
  // CHECK-NEXT:   %[[PART_ID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK:        %[[SLICE:.*]] = stablehlo.dynamic_slice %arg1
  // CHECK:        sdy.return %[[SLICE]] : tensor<4x16xf32>
  // CHECK-NEXT: } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK: return %[[MANUAL]] : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @skip_sharding_custom_call
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @skip_sharding_custom_call(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NOT: sdy.manual_computation
  // CHECK: %[[CC:.*]] = stablehlo.custom_call @Sharding(%[[ARG0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK: return %[[CC]] : tensor<8x16xf32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

