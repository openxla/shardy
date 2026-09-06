// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning="filter=dot,constant,reshard,all_gather,all_slice,concatenate,convolution,while,call,if" | FileCheck %s

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

  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[LHS]], %[[RHS]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg2: tensor<3x32xf32>, %arg3: tensor<32x16xf32>) {
  // CHECK:        %[[INNER_SLICE:.*]] = stablehlo.slice %arg2 [0:3, 0:32] : (tensor<3x32xf32>) -> tensor<3x32xf32>
  // CHECK-NEXT:   %[[LOCAL_DOT:.*]] = stablehlo.dot %[[INNER_SLICE]], %arg3 : (tensor<3x32xf32>, tensor<32x16xf32>) -> tensor<3x16xf32>
  // CHECK:        sdy.return %{{.*}} : tensor<3x16xf32>
  // CHECK-NEXT: } : (tensor<6x32xf32>, tensor<32x16xf32>) -> tensor<6x16xf32>
  // CHECK-NEXT: %[[SLICED_RES:.*]] = stablehlo.slice %[[MANUAL]] [0:5, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<6x16xf32>) -> tensor<5x16xf32>
  %dot = stablehlo.dot %sliced_lhs, %rhs {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<5x32xf32>, tensor<32x16xf32>) -> tensor<5x16xf32>

  // CHECK:      %[[MANUAL_RESHARD:.*]] = sdy.manual_computation(%[[MANUAL]])
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

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @selective_indivisible_reshard_sharded_to_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x5xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<5x5xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
func.func @selective_indivisible_reshard_sharded_to_sharded(
    %arg0: tensor<5x5xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<5x5xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:      %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 1], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<5x5xf32>, tensor<f32>) -> tensor<6x6xf32>
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[PAD]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{}, {"y"}]>]
  // CHECK-SAME:   manual_axes={"x", "y"} (%arg1: tensor<3x6xf32>) {
  // CHECK-NOT:    stablehlo.slice
  // CHECK:        %[[DYN_SLICE:.*]] = stablehlo.dynamic_slice %arg1
  // CHECK:        %[[ALL_GATHER:.*]] = "stablehlo.all_gather"(%[[DYN_SLICE]])
  // CHECK:        sdy.return %[[ALL_GATHER]] : tensor<6x3xf32>
  // CHECK-NEXT: } : (tensor<6x6xf32>) -> tensor<6x6xf32>
  // CHECK-NEXT: %[[SLICED:.*]] = stablehlo.slice %[[MANUAL]] [0:5, 0:5] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<6x6xf32>) -> tensor<5x5xf32>
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<5x5xf32>
  // CHECK: return %[[SLICED]] : tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
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

// -----

sdy.mesh @mesh = <["x"=6]>

// CHECK-LABEL: func @selective_indivisible_subaxis_reshard
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x":(1)2}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"x":(2)3}]>})
func.func @selective_indivisible_subaxis_reshard(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x":(1)2}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"x":(2)3}]>}) {
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:      %[[PAD0:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [0, 4], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x":(1)2}]>]>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<8x12xf32>
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[PAD0]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{}, {"x":(1)2}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x":(1)2}, {"x":(2)3}]>]
  // CHECK-SAME:   manual_axes={"x"}
  // CHECK:      %[[SLICE:.*]] = stablehlo.slice %[[MANUAL]] [0:8, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"x":(2)3}]>]>} : (tensor<8x12xf32>) -> tensor<8x8xf32>
  // CHECK:      return %[[SLICE]] : tensor<8x8xf32>
  %0 = sdy.reshard %arg0 <@mesh, [{"x":(1)2}, {"x":(2)3}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @indivisible_sdy_constant
// CHECK-SAME: () -> (tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @indivisible_sdy_constant() -> (tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation()
  // CHECK-SAME:   in_shardings=[]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} () {
  // CHECK-NEXT:   %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<3x16xf32>
  // CHECK-NEXT:   sdy.return %[[CST]] : tensor<3x16xf32>
  // CHECK-NEXT: } : () -> tensor<6x16xf32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[MANUAL]] [0:5, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<6x16xf32>) -> tensor<5x16xf32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<5x16xf32>
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} dense<1.000000e+00> : tensor<5x16xf32>
  return %0 : tensor<5x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @preserve_unreduced_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}], unreduced={"y"}>})
func.func @preserve_unreduced_axes(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}], unreduced={"y"}>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[ARG0]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}], unreduced={"y"}>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{}, {"x"}], unreduced={"y"}>]
  // CHECK-SAME:   manual_axes={"x", "y"} (%arg1: tensor<4x16xf32>) {
  // CHECK-NEXT:   %[[A2A:.*]] = "stablehlo.all_to_all"(%arg1)
  // CHECK:        sdy.return %[[A2A]] : tensor<8x8xf32>
  // CHECK-NEXT: } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[MANUAL]] : tensor<8x16xf32>
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"x"}], unreduced={"y"}> : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh2 = <["x"=2]>

// CHECK-LABEL: func @complex_indivisible_padding
func.func @complex_indivisible_padding(%arg0: tensor<5x16xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh2, [{"x"}, {}]>},
                                       %arg1: tensor<16x16xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh2, [{}, {}]>})
    -> (tensor<5x16xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh2, [{"x"}, {}]>}) {
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %[[ZERO]], low = [0, 0], high = [1, 0], interior = [0, 0]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"x"}, {}]>]>} : (tensor<5x16xcomplex<f32>>, tensor<16x16xcomplex<f32>>) -> tensor<5x16xcomplex<f32>>
  return %0 : tensor<5x16xcomplex<f32>>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @concatenate_sharded_concat_dim
func.func @concatenate_sharded_concat_dim(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>},
                                          %arg1: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}]>, <@mesh, [{"x"}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg2: tensor<2xf32>, %arg3: tensor<1xf32>) {
  // CHECK:        %[[AG0:.*]] = "stablehlo.all_gather"(%arg2)
  // CHECK:        %[[AG1:.*]] = "stablehlo.all_gather"(%arg3)
  // CHECK:        %[[CONCAT:.*]] = stablehlo.concatenate %[[AG0]], %[[AG1]], dim = 0 : (tensor<4xf32>, tensor<2xf32>) -> tensor<6xf32>
  // CHECK:        %[[SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], {{.*}}, sizes = [3] : (tensor<6xf32>, tensor<i64>) -> tensor<3xf32>
  // CHECK:        sdy.return %[[SLICE]] : tensor<3xf32>
  // CHECK-NEXT: } : (tensor<4xf32>, tensor<2xf32>) -> tensor<6xf32>
  // CHECK-NEXT: return %[[MANUAL]] : tensor<6xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4xf32>, tensor<2xf32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @concatenate_indivisible_sharded_concat_dim
func.func @concatenate_indivisible_sharded_concat_dim(%arg0: tensor<5xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>},
                                                      %arg1: tensor<1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK:      %[[PAD0:.*]] = stablehlo.pad %arg0, {{.*}}, low = [0], high = [1], interior = [0]
  // CHECK:      %[[PAD1:.*]] = stablehlo.pad %arg1, {{.*}}, low = [0], high = [1], interior = [0]
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[PAD0]], %[[PAD1]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}]>, <@mesh, [{"x"}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg2: tensor<3xf32>, %arg3: tensor<1xf32>) {
  // CHECK:        %[[IN_SLICE0:.*]] = stablehlo.slice %arg2 [0:3] : (tensor<3xf32>) -> tensor<3xf32>
  // CHECK:        %[[AG0:.*]] = "stablehlo.all_gather"(%[[IN_SLICE0]]) {{.*}} : (tensor<3xf32>) -> tensor<6xf32>
  // CHECK:        %[[SLICE0:.*]] = stablehlo.slice %[[AG0]] [0:5] : (tensor<6xf32>) -> tensor<5xf32>
  // CHECK:        %[[IN_SLICE1:.*]] = stablehlo.slice %arg3 [0:1] : (tensor<1xf32>) -> tensor<1xf32>
  // CHECK:        %[[AG1:.*]] = "stablehlo.all_gather"(%[[IN_SLICE1]]) {{.*}} : (tensor<1xf32>) -> tensor<2xf32>
  // CHECK:        %[[SLICE1:.*]] = stablehlo.slice %[[AG1]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
  // CHECK:        %[[CONCAT:.*]] = stablehlo.concatenate %[[SLICE0]], %[[SLICE1]], dim = 0 : (tensor<5xf32>, tensor<1xf32>) -> tensor<6xf32>
  // CHECK:        %[[SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], {{.*}}, sizes = [3] : (tensor<6xf32>, tensor<i64>) -> tensor<3xf32>
  // CHECK:        sdy.return %[[SLICE]] : tensor<3xf32>
  // CHECK-NEXT: } : (tensor<6xf32>, tensor<2xf32>) -> tensor<6xf32>
  // CHECK-NEXT: return %[[MANUAL]] : tensor<6xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<5xf32>, tensor<1xf32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @conv_dual_semantics_with_trailing_all_reduce
func.func @conv_dual_semantics_with_trailing_all_reduce(
    %arg0: tensor<1x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>},
    %arg1: tensor<6x4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>})
    -> (tensor<1x3x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME:   in_shardings=[<@mesh, [{}, {"x"}, {}]>, <@mesh, [{"x"}, {}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{}, {}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg2: tensor<1x4x4xf32>, %arg3: tensor<3x4x4xf32>) {
  // CHECK-NEXT:   %[[AG0:.*]] = "stablehlo.all_gather"(%arg2)
  // CHECK-NEXT:   %[[AG1:.*]] = "stablehlo.all_gather"(%arg3)
  // CHECK-NEXT:   %[[CONV:.*]] = stablehlo.convolution(%[[AG0]], %[[AG1]])
  // CHECK-NOT:    all_reduce
  // CHECK:        sdy.return %[[CONV]] : tensor<1x3x4xf32>
  // CHECK-NEXT: } : (tensor<1x8x4xf32>, tensor<6x4x4xf32>) -> tensor<1x3x4xf32>
  // CHECK-NOT:  sdy.all_reduce
  // CHECK-NEXT: return %[[MANUAL]] : tensor<1x3x4xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {stride = [1], pad = [[0, 0]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}], unreduced={"x"}>]>
    } : (tensor<1x8x4xf32>, tensor<6x4x4xf32>) -> tensor<1x3x4xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh, [{}, {}, {}]> : tensor<1x3x4xf32>
  return %1 : tensor<1x3x4xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @conv_pure_reduction_with_trailing_all_reduce
func.func @conv_pure_reduction_with_trailing_all_reduce(
    %arg0: tensor<1x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>},
    %arg1: tensor<8x4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>})
    -> (tensor<1x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME:   in_shardings=[<@mesh, [{}, {"x"}, {}]>, <@mesh, [{"x"}, {}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{}, {}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg2: tensor<1x4x4xf32>, %arg3: tensor<4x4x4xf32>) {
  // CHECK-NEXT:   %[[CONV:.*]] = stablehlo.convolution(%arg2, %arg3)
  // CHECK-NEXT:   %[[AR:.*]] = "stablehlo.all_reduce"(%[[CONV]])
  // CHECK:        sdy.return %[[AR]] : tensor<1x1x4xf32>
  // CHECK-NEXT: } : (tensor<1x8x4xf32>, tensor<8x4x4xf32>) -> tensor<1x1x4xf32>
  // CHECK-NOT:  sdy.all_reduce
  // CHECK-NEXT: return %[[MANUAL]] : tensor<1x1x4xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {stride = [1], pad = [[0, 0]]}
    {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}], unreduced={"x"}>]>
    } : (tensor<1x8x4xf32>, tensor<8x4x4xf32>) -> tensor<1x1x4xf32>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh, [{}, {}, {}]> : tensor<1x1x4xf32>
  return %1 : tensor<1x1x4xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @while_with_implicit_capture
// CHECK-SAME: (%[[INIT_I:.*]]: tensor<i32>,
// CHECK-SAME:  %[[INIT_VAL:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:  %[[OUTSIDE:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @while_with_implicit_capture(
    %init_i: tensor<i32>,
    %init_val: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %outside: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK:      %[[MANUAL:.*]]:2 = sdy.manual_computation(%[[INIT_I]], %[[INIT_VAL]], %[[OUTSIDE]])
  // CHECK-SAME:   in_shardings=[<@mesh, []>, <@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, []>, <@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%[[LOCAL_I:.*]]: tensor<i32>, %[[LOCAL_VAL:.*]]: tensor<4x16xf32>, %[[LOCAL_OUTSIDE:.*]]: tensor<4x16xf32>) {
  // CHECK:        %[[WHILE:.*]]:2 = stablehlo.while(%[[ITER_I:.*]] = %[[LOCAL_I]], %[[ITER_VAL:.*]] = %[[LOCAL_VAL]]) : tensor<i32>, tensor<4x16xf32>
  // CHECK:          %[[ADD:.*]] = stablehlo.add %[[ITER_VAL]], %[[LOCAL_OUTSIDE]] : tensor<4x16xf32>
  // CHECK-NEXT:     stablehlo.return %{{.*}}, %[[ADD]] : tensor<i32>, tensor<4x16xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT:   sdy.return %[[WHILE]]#0, %[[WHILE]]#1 : tensor<i32>, tensor<4x16xf32>
  // CHECK-NEXT: } : (tensor<i32>, tensor<8x16xf32>, tensor<8x16xf32>) -> (tensor<i32>, tensor<8x16xf32>)
  %0:2 = stablehlo.while(%iter_i = %init_i, %iter_val = %init_val) : tensor<i32>, tensor<8x16xf32>
      attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh, []>, <@mesh, [{"x"}, {}]>]>}
    cond {
      %limit = stablehlo.constant dense<10> : tensor<i32>
      %cmp = stablehlo.compare LT, %iter_i, %limit : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %step = stablehlo.constant dense<1> : tensor<i32>
      %next_i = stablehlo.add %iter_i, %step : tensor<i32>
      %add = stablehlo.add %iter_val, %outside {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
      stablehlo.return %next_i, %add : tensor<i32>, tensor<8x16xf32>
    }
  return %0#1 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func private @callee(
// CHECK-SAME: %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func private @callee(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @call_with_callee
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @call_with_callee(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[ARG0]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%[[LOCAL_ARG:.*]]: tensor<4x16xf32>) {
  // CHECK-NEXT:   %[[CALL:.*]] = func.call @callee_0(%[[LOCAL_ARG]]) : (tensor<4x16xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT:   sdy.return %[[CALL]] : tensor<4x16xf32>
  // CHECK-NEXT: } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = func.call @callee(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func private @callee_0(%arg0: tensor<4x16xf32>) -> tensor<4x16xf32> {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<4x16xf32>
// CHECK-NEXT:    return %[[ADD]] : tensor<4x16xf32>

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @if_with_implicit_capture
// CHECK-SAME: (%[[PRED:.*]]: tensor<i1>,
// CHECK-SAME:  %[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @if_with_implicit_capture(
    %pred: tensor<i1>,
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[PRED]], %[[ARG0]], %[[ARG1]])
  // CHECK-SAME:   manual_axes={"x"} (%[[LOCAL_PRED:.*]]: tensor<i1>, %[[LOCAL_ARG0:.*]]: tensor<4x16xf32>, %[[LOCAL_ARG1:.*]]: tensor<4x16xf32>) {
  // CHECK-NEXT:   %[[IF:.*]] = "stablehlo.if"(%[[LOCAL_PRED]]) ({
  // CHECK-NEXT:     %[[ADD:.*]] = stablehlo.add %[[LOCAL_ARG0]], %[[LOCAL_ARG1]] : tensor<4x16xf32>
  // CHECK-NEXT:     stablehlo.return %[[ADD]] : tensor<4x16xf32>
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     %[[SUB:.*]] = stablehlo.subtract %[[LOCAL_ARG0]], %[[LOCAL_ARG1]] : tensor<4x16xf32>
  // CHECK-NEXT:     stablehlo.return %[[SUB]] : tensor<4x16xf32>
  // CHECK-NEXT:   }) : (tensor<i1>) -> tensor<4x16xf32>
  // CHECK-NEXT:   sdy.return %[[IF]] : tensor<4x16xf32>
  // CHECK-NEXT: } : (tensor<i1>, tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = "stablehlo.if"(%pred) ({
    %add = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
    stablehlo.return %add : tensor<8x16xf32>
  }, {
    %sub = stablehlo.subtract %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
    stablehlo.return %sub : tensor<8x16xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<i1>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// Tests indivisible op input (%init_val: 5x16), indivisible implicit capture
// (%outside: 5x16), and indivisible op output (5x16) on stablehlo.while.
// CHECK-LABEL: func @while_with_indivisible_capture_input_and_output
// CHECK-SAME: (%[[INIT_I:.*]]: tensor<i32>,
// CHECK-SAME:  %[[INIT_VAL:.*]]: tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:  %[[OUTSIDE:.*]]: tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @while_with_indivisible_capture_input_and_output(
    %init_i: tensor<i32>,
    %init_val: tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %outside: tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // (1) Pad high 0 to make both indivisible input and indivisible implicit capture divisible (5x16 -> 6x16):
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:      %[[PAD_VAL:.*]] = stablehlo.pad %[[INIT_VAL]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
  // CHECK:      %[[CST2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:      %[[PAD_OUTSIDE:.*]] = stablehlo.pad %[[OUTSIDE]], %[[CST2]], low = [0, 0], high = [1, 0], interior = [0, 0]
  // CHECK:      %[[MANUAL:.*]]:2 = sdy.manual_computation(%[[INIT_I]], %[[PAD_VAL]], %[[PAD_OUTSIDE]])
  // CHECK-SAME:   in_shardings=[<@mesh, []>, <@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, []>, <@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%[[LOCAL_I:.*]]: tensor<i32>, %[[LOCAL_VAL:.*]]: tensor<3x16xf32>, %[[LOCAL_OUTSIDE:.*]]: tensor<3x16xf32>) {
  // (2)-(4) Inside manual_computation, local shard shape is 3x16 (from padded 6x16 / 2):
  // CHECK:        %[[SLICE_OUTSIDE:.*]] = stablehlo.slice %[[LOCAL_OUTSIDE]] [0:3, 0:16] : (tensor<3x16xf32>) -> tensor<3x16xf32>
  // CHECK:        %[[SLICE_VAL:.*]] = stablehlo.slice %[[LOCAL_VAL]] [0:3, 0:16] : (tensor<3x16xf32>) -> tensor<3x16xf32>
  // CHECK:        %[[WHILE:.*]]:2 = stablehlo.while(%[[ITER_I:.*]] = %[[LOCAL_I]], %[[ITER_VAL:.*]] = %[[SLICE_VAL]]) : tensor<i32>, tensor<3x16xf32>
  // CHECK:        sdy.return %[[WHILE]]#0, %{{.*}} : tensor<i32>, tensor<3x16xf32>
  // CHECK-NEXT: } : (tensor<i32>, tensor<6x16xf32>, tensor<6x16xf32>) -> (tensor<i32>, tensor<6x16xf32>)
  // (5) In outer region after manual_computation, slice result back to indivisible shape (6x16 -> 5x16):
  // CHECK-NEXT: %[[SLICE_RES:.*]] = stablehlo.slice %[[MANUAL]]#1 [0:5, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<6x16xf32>) -> tensor<5x16xf32>
  // CHECK-NEXT: return %[[SLICE_RES]] : tensor<5x16xf32>
  %0:2 = stablehlo.while(%iter_i = %init_i, %iter_val = %init_val) : tensor<i32>, tensor<5x16xf32>
      attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh, []>, <@mesh, [{"x"}, {}]>]>}
    cond {
      %limit = stablehlo.constant dense<10> : tensor<i32>
      %cmp = stablehlo.compare LT, %iter_i, %limit : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %step = stablehlo.constant dense<1> : tensor<i32>
      %next_i = stablehlo.add %iter_i, %step : tensor<i32>
      %add = stablehlo.add %iter_val, %outside {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<5x16xf32>
      stablehlo.return %next_i, %add : tensor<i32>, tensor<5x16xf32>
    }
  return %0#1 : tensor<5x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// Tests indivisible implicit capture (%lhs: 5x16, %rhs: 5x16) and indivisible op
// output (5x16) on stablehlo.if.
// CHECK-LABEL: func @if_with_indivisible_capture_and_output
// CHECK-SAME: (%[[PRED:.*]]: tensor<i1>,
// CHECK-SAME:  %[[LHS:.*]]: tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:  %[[RHS:.*]]: tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @if_with_indivisible_capture_and_output(
    %pred: tensor<i1>,
    %lhs: tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %rhs: tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:      %[[PAD_LHS:.*]] = stablehlo.pad %[[LHS]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
  // CHECK:      %[[CST2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:      %[[PAD_RHS:.*]] = stablehlo.pad %[[RHS]], %[[CST2]], low = [0, 0], high = [1, 0], interior = [0, 0]
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[PRED]], %[[PAD_LHS]], %[[PAD_RHS]])
  // CHECK-SAME:   in_shardings=[<@mesh, []>, <@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%[[LOCAL_PRED:.*]]: tensor<i1>, %[[LOCAL_LHS:.*]]: tensor<3x16xf32>, %[[LOCAL_RHS:.*]]: tensor<3x16xf32>) {
  // CHECK:        %[[SLICE_LHS:.*]] = stablehlo.slice %[[LOCAL_LHS]] [0:3, 0:16] : (tensor<3x16xf32>) -> tensor<3x16xf32>
  // CHECK:        %[[SLICE_RHS:.*]] = stablehlo.slice %[[LOCAL_RHS]] [0:3, 0:16] : (tensor<3x16xf32>) -> tensor<3x16xf32>
  // CHECK:        %[[IF:.*]] = "stablehlo.if"(%[[LOCAL_PRED]]) ({
  // CHECK:          %[[ADD:.*]] = stablehlo.add %[[SLICE_LHS]], %[[SLICE_RHS]] : tensor<3x16xf32>
  // CHECK:          stablehlo.return %[[ADD]] : tensor<3x16xf32>
  // CHECK:        }, {
  // CHECK:          %[[SUB:.*]] = stablehlo.subtract %[[SLICE_LHS]], %[[SLICE_RHS]] : tensor<3x16xf32>
  // CHECK:          stablehlo.return %[[SUB]] : tensor<3x16xf32>
  // CHECK:        }) : (tensor<i1>) -> tensor<3x16xf32>
  // CHECK:        sdy.return %{{.*}} : tensor<3x16xf32>
  // CHECK-NEXT: } : (tensor<i1>, tensor<6x16xf32>, tensor<6x16xf32>) -> tensor<6x16xf32>
  // CHECK-NEXT: %[[SLICE_RES:.*]] = stablehlo.slice %[[MANUAL]] [0:5, 0:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<6x16xf32>) -> tensor<5x16xf32>
  // CHECK-NEXT: return %[[SLICE_RES]] : tensor<5x16xf32>
  %0 = "stablehlo.if"(%pred) ({
    %add = stablehlo.add %lhs, %rhs {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<5x16xf32>
    stablehlo.return %add : tensor<5x16xf32>
  }, {
    %sub = stablehlo.subtract %lhs, %rhs {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<5x16xf32>
    stablehlo.return %sub : tensor<5x16xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<i1>) -> tensor<5x16xf32>
  return %0 : tensor<5x16xf32>
}
