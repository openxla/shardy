// RUN: sdy_opt %s -sdy-basic-propagate -verify-diagnostics 2>&1 | FileCheck %s

sdy.mesh @mesh_a_3 = <["a"=3]>
sdy.mesh @mesh_a_6 = <["a"=6]>
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_3 = <["a"=2, "b"=3]>
sdy.mesh @mesh_a_3_b_3 = <["a"=3, "b"=3]>
sdy.mesh @mesh_a_4_b_2 = <["a"=4, "b"=2]>
sdy.mesh @mesh_a_4_b_4 = <["a"=4, "b"=4]>
sdy.mesh @mesh_a_6_b_2 = <["a"=6, "b"=2]>
sdy.mesh @mesh_a_16_b_2 = <["a"=16, "b"=2]>
sdy.mesh @mesh_a_1_b_2_c_1 = <["a"=1, "b"=2, "c"=1]>
sdy.mesh @mesh_a_1_b_2_c_1_d_2 = <["a"=1, "b"=2, "c"=1, "d"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>
sdy.mesh @mesh_a_4_b_2_c_2 = <["a"=4, "b"=2, "c"=2]>
sdy.mesh @mesh_a_2_b_3_c_2 = <["a"=2, "b"=3, "c"=2]>
sdy.mesh @mesh_a_2_b_3_c_2_d_2 = <["a"=2, "b"=3, "c"=2, "d"=2]>
sdy.mesh @mesh_a_3_another = <["a"=3]>

// CHECK-LABEL: func @simple(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>},
// CHECK-SAME:      %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", ?}, {?}]>})
// CHECK-SAME:  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>}) {
func.func @simple(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
                  %arg1: tensor<8x8xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: stablehlo.dot_general %[[ADD]], %arg2
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {?}]>]>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @pointwise_size_zero_dim(
// CHECK-SAME:      %arg0: tensor<8x0xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
// CHECK-SAME:  -> (tensor<8x0xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>}) {
func.func @pointwise_size_zero_dim(%arg0: tensor<8x0xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>}) -> tensor<8x0xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {?}]>]>}
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x0xf32>
  return %0 : tensor<8x0xf32>
}

// CHECK-LABEL: func @propagate_to_multi_result_op
func.func @propagate_to_multi_result_op(%arg0: tensor<4x64x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{}, {}, {"b"}]>},
                                        %arg1: tensor<4x64x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{}, {}, {"b"}]>})
    -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  // CHECK-NEXT: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
  // CHECK-NEXT: stablehlo.reduce(%arg0 init: %[[CONST]]), (%arg1 init: %[[CONST]]) across dimensions = [1]
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}, {"b", ?}]>, <@mesh_a_2_b_2, [{?}, {"b", ?}]>]>}
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1] :
    (tensor<4x64x8xf32>, tensor<4x64x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
      %2 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %3 = stablehlo.add %arg3, %arg5 : tensor<f32>
      stablehlo.return %2, %3 : tensor<f32>, tensor<f32>
    }
  return %1#0, %1#1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @propagate_from_multi_result_op
// CHECK-SAME:      %arg0: tensor<4x64x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {?}, {"b", ?}]>},
// CHECK-SAME:      %arg1: tensor<4x64x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {?}, {"b", ?}]>})
// CHECK-SAME:  -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b", ?}]>}, tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b", ?}]>}) {
func.func @propagate_from_multi_result_op(%arg0: tensor<4x64x8xf32>, %arg1: tensor<4x64x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}, {"b", ?}]>, <@mesh_a_2_b_2, [{?}, {"b", ?}]>]>} :
    (tensor<4x64x8xf32>, tensor<4x64x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
      %2 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %3 = stablehlo.add %arg3, %arg5 : tensor<f32>
      stablehlo.return %2, %3 : tensor<f32>, tensor<f32>
    }
  return %1#0, %1#1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @closed_dim(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{}, {"b"}]>})
func.func @closed_dim(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>},
                      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{}, {"b"}]>}) -> tensor<8x16xf32> {
  // CHECK-NEXT: stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{}, {"b", ?}]>]>}
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{}, {?}]>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @propagate_from_sharding_constraint(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>}) {
func.func @propagate_from_sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[SHARDING_CONSTRAINT:.*]] = sdy.sharding_constraint %arg0 <@mesh_a_2_b_2, [{"a"}, {?}]>
  // CHECK-NEXT: stablehlo.add %[[SHARDING_CONSTRAINT]], %[[SHARDING_CONSTRAINT]]
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {?}]>]>}
  %0 = sdy.sharding_constraint %arg0 <@mesh_a_2_b_2, [{"a"}, {?}]> : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @propagate_to_sharding_constraint(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
func.func @propagate_to_sharding_constraint(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: sdy.sharding_constraint %arg0 <@mesh_a_2_b_2, [{"a", ?}, {?}]>
  %0 = sdy.sharding_constraint %arg0 <@mesh_a_2_b_2, [{?}, {?}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @propagation_barrier_backward_just_arg_sharding(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>})
// CHECK-SAME:  -> tensor<8x8xf32> {
func.func @propagation_barrier_backward_just_arg_sharding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8x8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @propagation_barrier_backward_just_result_sharding(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>}) {
func.func @propagation_barrier_backward_just_result_sharding(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>}) {
  // CHECK-NEXT: sdy.propagation_barrier %arg0 allowed_direction=BACKWARD {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>]>} : tensor<8x8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @propagation_barrier_backward_two_shardings(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {"c", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>}) {
func.func @propagation_barrier_backward_two_shardings(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a","b",?}, {?}]>}) {
  // CHECK-NEXT: sdy.propagation_barrier %arg0 allowed_direction=BACKWARD {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>]>} : tensor<8x8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// Do not push the "c" in dim 1 through backwards.
//
// CHECK-LABEL: func @propagation_barrier_backward_multiple_uses(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {"c", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>},
// CHECK-SAME:      tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>})
func.func @propagation_barrier_backward_multiple_uses(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a","b",?}, {?}]>}, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[ADD0:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[BARRIER:.*]] = sdy.propagation_barrier %[[ADD0]] allowed_direction=BACKWARD {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[ADD1:.*]] = stablehlo.add %[[BARRIER]], %[[BARRIER]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>]>} : tensor<8x8xf32
  // CHECK-NEXT: %[[ADD2:.*]] = stablehlo.add %[[BARRIER]], %[[BARRIER]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>]>} : tensor<8x8xf32
  // CHECK-NEXT: return %[[ADD1]], %[[ADD2]] : tensor<8x8xf32>, tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %1 = sdy.propagation_barrier %0 allowed_direction=BACKWARD : tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x8xf32>
  %3 = stablehlo.add %1, %1 : tensor<8x8xf32>
  return %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @propagation_barrier_forward_just_arg_sharding(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>}) {
func.func @propagation_barrier_forward_just_arg_sharding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: sdy.propagation_barrier %arg0 allowed_direction=FORWARD {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @propagation_barrier_forward_just_result_sharding(
// CHECK-SAME:      %arg0: tensor<8x8xf32>)
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>}) {
func.func @propagation_barrier_forward_just_result_sharding(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>}) {
  // CHECK-NEXT: sdy.propagation_barrier %arg0 allowed_direction=FORWARD {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>]>} : tensor<8x8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @propagation_barrier_forward_two_shardings(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {"c", ?}]>}) {
func.func @propagation_barrier_forward_two_shardings(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a","b",?}, {?}]>}) {
  // CHECK-NEXT: sdy.propagation_barrier %arg0 allowed_direction=FORWARD {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// Do not push the "b" in dim 0 through forwards.
//
// CHECK-LABEL: func @propagation_barrier_forward_multiple_uses(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {"c", ?}]>},
// CHECK-SAME:      tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {"c", ?}]>})
func.func @propagation_barrier_forward_multiple_uses(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a","b",?}, {?}]>}, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[ADD0:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[BARRIER:.*]] = sdy.propagation_barrier %[[ADD0]] allowed_direction=FORWARD {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[ADD1:.*]] = stablehlo.add %[[BARRIER]], %[[BARRIER]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {"c", ?}]>]>} : tensor<8x8xf32
  // CHECK-NEXT: %[[ADD2:.*]] = stablehlo.add %[[BARRIER]], %[[BARRIER]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {"c", ?}]>]>} : tensor<8x8xf32
  // CHECK-NEXT: return %[[ADD1]], %[[ADD2]] : tensor<8x8xf32>, tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %1 = sdy.propagation_barrier %0 allowed_direction=FORWARD : tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x8xf32>
  %3 = stablehlo.add %1, %1 : tensor<8x8xf32>
  return %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @propagation_barrier_none_two_shardings(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>}) {
func.func @propagation_barrier_none_two_shardings(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", ?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a","b",?}, {?}]>}) {
  // CHECK-NEXT: sdy.propagation_barrier %arg0 allowed_direction=NONE {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>]>} : tensor<8x8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=NONE : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @multi_axes(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", "a"}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", "a", ?}, {}]>},
// CHECK-SAME:      %arg2: tensor<8x16xf32>)
func.func @multi_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", "a"}, {}]>},
                      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", ?}, {}]>},
                      %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"b", "a", ?}, {?}]>]>}
  // CHECK-NEXT: stablehlo.dot_general %[[ADD]], %arg2
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"b", "a", ?}, {?}]>]>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @multi_axes_conflict
func.func @multi_axes_conflict(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", "a"}, {}]>},
                               %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", "c"}, {}]>},
                               %arg2: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}, {}]>})
                              -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"b", ?}, {?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: stablehlo.add %arg0, %arg2 : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.add %arg0, %arg2 : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @multi_axes_some_axes_incompatible(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>})
func.func @multi_axes_some_axes_incompatible(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}, {}]>},
                                             %arg1: tensor<8x8xf32>)
                              -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>} : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}, {"b", ?}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @multi_axes_all_axes_incompatible(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {?}], replicated={"a"}>})
func.func @multi_axes_all_axes_incompatible(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}, {}]>},
                                            %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {?}], replicated={"a"}>})
                              -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}, {"b", ?}]>]>} : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}, {"b", ?}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @multi_axes_compatible_prefix(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a":(1)4, ?}, {"a":(4)2, ?}]>})
func.func @multi_axes_compatible_prefix(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a", ?}, {?}]>},
                                        %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a":(1)2, ?}, {"a":(4)2, ?}]>})
                              -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a":(1)4, ?}, {"b", ?}]>]>} : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{?}, {"b", ?}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @pointwise_size_one_axes(
// CHECK-SAME:      %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_1_b_2_c_1_d_2, [{"d", ?}, {"a", "b", "c"}]>})
func.func @pointwise_size_one_axes(
    %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_1_b_2_c_1_d_2, [{?}, {"a", "b", "c"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_1_b_2_c_1_d_2, [{"d"}, {"a", "b", "c", ?}]>]>}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_1_b_2_c_1_d_2, [{"d"}, {?}]>]>} : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @slice_then_concat(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}, {}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}, {?}]>}) {
func.func @slice_then_concat(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[SLICE_0:.*]] = stablehlo.slice %arg0 [0:1, 0:8]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}, {?}]>]>}
  // CHECK-NEXT: %[[SLICE_1:.*]] = stablehlo.slice %arg0 [1:8, 0:8]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}, {?}]>]>}
  // CHECK-NEXT: %[[CONCAT:.*]] = stablehlo.concatenate %[[SLICE_1]], %[[SLICE_0]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}, {?}]>]>}
  // CHECK-NEXT: return %[[CONCAT]]
  %0 = stablehlo.slice %arg0 [0:1, 0:8] : (tensor<8x8xf32>) -> tensor<1x8xf32>
  %1 = stablehlo.slice %arg0 [1:8, 0:8] : (tensor<8x8xf32>) -> tensor<7x8xf32>
  %2 = stablehlo.concatenate %1, %0, dim = 0 : (tensor<7x8xf32>, tensor<1x8xf32>) -> tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// CHECK-LABEL: func @reshape_size_one_axes(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_1_b_2_c_1, [{"a", "b", "c"}]>})
func.func @reshape_size_one_axes(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_1_b_2_c_1, [{"a", "b", "c"}]>}) -> tensor<2x1x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_1_b_2_c_1, [{"a", "b", "c", ?}, {?}, {?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x1x4xf32>
  return %0 : tensor<2x1x4xf32>
}

// CHECK-LABEL: func @reshape_merge_dim_only_major_most_factor_sharded
func.func @reshape_merge_dim_only_major_most_factor_sharded(
    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @reshape_merge_dim_major_factor_fully_sharded
func.func @reshape_merge_dim_major_factor_fully_sharded(
    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @reshape_merge_dim_major_factor_partially_sharded
func.func @reshape_merge_dim_major_factor_partially_sharded(
    %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) -> tensor<16xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}]>]>}
  // expected-warning@+1 {{can't propagate sharding as strided view is needed}}
  %0 = stablehlo.reshape %arg0 : (tensor<4x4xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func @reshape_split_dim_single_axis_fully_shards_major_factor
func.func @reshape_split_dim_single_axis_fully_shards_major_factor(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", "a"}]>}) -> tensor<2x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"b", ?}, {"a", ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @reshape_split_dim_only_major_most_factor_sharded
func.func @reshape_split_dim_only_major_most_factor_sharded(
    %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", "a"}]>}) -> tensor<4x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"b", "a", ?}, {?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @reshape_split_dim_single_axis_shards_both_factors
func.func @reshape_split_dim_single_axis_shards_both_factors(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2, [{"a"}]>}) -> tensor<2x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4_b_2, [{"a":(1)2, ?}, {"a":(2)2, ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @reshape_split_dim_twice
func.func @reshape_split_dim_twice(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a"}]>}) -> tensor<2x4x4xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a":(1)8, ?}, {"a":(8)2, ?}]>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[RESHAPE_0]], %[[RESHAPE_0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a":(1)8, ?}, {"a":(8)2, ?}]>]>}
  // CHECK-NEXT: stablehlo.reshape %[[ADD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a":(1)2, ?}, {"a":(2)4, ?}, {"a":(8)2, ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<32xf32>) -> tensor<8x4xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x4xf32>
  %2 = stablehlo.reshape %1 : (tensor<8x4xf32>) -> tensor<2x4x4xf32>
  return %2 : tensor<2x4x4xf32>
}

// CHECK-LABEL: func @reshape_split_dim_then_merge
func.func @reshape_split_dim_then_merge(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a"}]>}) -> tensor<32xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a":(1)4, ?}, {"a":(4)2, ?}, {"a":(8)2, ?}]>]>}
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[RESHAPE_0]], %[[RESHAPE_0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a":(1)4, ?}, {"a":(4)2, ?}, {"a":(8)2, ?}]>]>}
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[ADD_0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a":(1)4, ?}, {"a":(4)4, ?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[RESHAPE_1]], %[[RESHAPE_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a":(1)4, ?}, {"a":(4)4, ?}]>]>}
  // CHECK-NEXT: stablehlo.reshape %[[ADD_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a", ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<32xf32>) -> tensor<4x2x4xf32>
  %1 = stablehlo.add %0, %0 : tensor<4x2x4xf32>
  %2 = stablehlo.reshape %1 : (tensor<4x2x4xf32>) -> tensor<4x8xf32>
  %3 = stablehlo.add %2, %2 : tensor<4x8xf32>
  %4 = stablehlo.reshape %3 : (tensor<4x8xf32>) -> tensor<32xf32>
  return %4 : tensor<32xf32>
}

// CHECK-LABEL: func @reshape_split_and_merge_dims
func.func @reshape_split_and_merge_dims(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2_c_2, [{"c", "a"}, {"b"}]>}) -> tensor<2x16xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4_b_2_c_2, [{"c", ?}, {"a", "b", ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<8x4xf32>) -> tensor<2x16xf32>
  return %0 : tensor<2x16xf32>
}

// CHECK-LABEL: func @reshape_size_zero_dim
func.func @reshape_size_zero_dim(%arg0: tensor<8x0xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>}) -> tensor<0x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0
  // CHECK-NOT:  sdy.sharding
  %0 = stablehlo.reshape %arg0 : (tensor<8x0xf32>) -> tensor<0x4xf32>
  return %0 : tensor<0x4xf32>
}

// CHECK-LABEL: func @propagate_full_to_sub_axis(
// CHECK-SAME:      %arg0: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2, [{"a", ?}, {}]>})
func.func @propagate_full_to_sub_axis(%arg0: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2, [{"a", ?}, {}]>}) -> tensor<32x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4_b_2, [{"a":(1)2, ?}, {}]>]>} : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
  return %0 : tensor<32x8xf32>
}

// CHECK-LABEL: func @single_factor_non_divisible
func.func @single_factor_non_divisible(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3, [{}, {"a"}]>}) -> tensor<2x4xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_3, [{?}, {"a", ?}]>]>}
  %0 = stablehlo.add %arg0, %arg0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @single_factor_overflows
func.func @single_factor_overflows(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2, [{"a"}, {}]>}) -> tensor<2x4xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4_b_2, [{"a", ?}, {?}]>]>}
  %0 = stablehlo.add %arg0, %arg0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @minor_most_factor_non_divisible
func.func @minor_most_factor_non_divisible(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_6, [{"a"}]>}) -> tensor<2x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_6, [{"a":(1)2, ?}, {"a":(2)3, ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @minor_most_factor_overflows
func.func @minor_most_factor_overflows(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a"}]>}) -> tensor<2x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a":(1)2, ?}, {"a":(2)8, ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @minor_most_factor_overflows_multiple_axes
func.func @minor_most_factor_overflows_multiple_axes(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4_b_4, [{"a", "b"}]>}) -> tensor<2x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4_b_4, [{"a":(1)2, ?}, {"a":(2)2, "b", ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @non_minor_most_factor_non_divisible(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3, [{"a"}]>})
func.func @non_minor_most_factor_non_divisible(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3, [{"a"}]>}) -> tensor<2x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0
  // CHECK-NOT:  sdy.sharding
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// NOTE: it's important to make sure that the sharding of %arg0 doesn't change,
// because "b" is added to the ShardingProjection as an overflow axis (see
// `FactorSharding`), that gets added back when creating the updated
// `TensorShardingAttr`.
// CHECK-LABEL: func @non_minor_most_factor_non_divisible_multiple_axes(
// CHECK-SAME:      %arg0: tensor<2x2x32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_3_c_2_d_2, [{"c"}, {"d", ?}, {"a", "b"}]>})
func.func @non_minor_most_factor_non_divisible_multiple_axes(
  %arg0: tensor<2x2x32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_3_c_2_d_2, [{"c"}, {?}, {"a", "b"}]>})
  -> tensor<2x2x8x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_3_c_2_d_2, [{"c", ?}, {"d"}, {"a", ?}, {?}]>]>}
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_3_c_2_d_2, [{?}, {"d"}, {?}, {?}]>]>} : (tensor<2x2x32xf32>) -> tensor<2x2x8x4xf32>
  return %0 : tensor<2x2x8x4xf32>
}

// NOTE: it's important to make sure that the sharding of %arg0 doesn't change,
// because "a":(2)3 is added to the ShardingProjection as an overflow axis (see
// `FactorSharding`), that gets added back when creating the updated
// `TensorShardingAttr`.
// CHECK-LABEL: func @non_minor_most_factor_non_divisible_sub_axis(
// CHECK-SAME:      %arg0: tensor<2x32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_6_b_2, [{"b", ?}, {"a"}]>})
func.func @non_minor_most_factor_non_divisible_sub_axis(
  %arg0: tensor<2x32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_6_b_2, [{?}, {"a"}]>})
  -> tensor<2x8x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_6_b_2, [{"b"}, {"a":(1)2, ?}, {?}]>]>}
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_6_b_2, [{"b"}, {?}, {?}]>]>} : (tensor<2x32xf32>) -> tensor<2x8x4xf32>
  return %0 : tensor<2x8x4xf32>
}

// This test verifies that "b" isn't propagated from the `stablehlo.reshape` to
// %arg0, even though "b" in %arg0 is an overflow axis (see `FactorSharding`).
// CHECK-LABEL: func @non_minor_most_factor_non_divisible_other_open_dim_unchanged(
// CHECK-SAME:      %arg0: tensor<3x32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_3, [{?}, {"a", "b", ?}]>})
func.func @non_minor_most_factor_non_divisible_other_open_dim_unchanged(
  %arg0: tensor<3x32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_3, [{?}, {"a", "b", ?}]>})
  -> tensor<3x8x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_3, [{"b"}, {"a", ?}, {?}]>]>}
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_3, [{"b"}, {?}, {?}]>]>} : (tensor<3x32xf32>) -> tensor<3x8x4xf32>
  return %0 : tensor<3x8x4xf32>
}

// This test verifies that "c" isn't propagated from the `stablehlo.reshape` to
// %arg0, even though its dimension is open, and that the dimension remains open.
// CHECK-LABEL: func @non_minor_most_factor_non_divisible_same_open_dim_unchanged(
// CHECK-SAME:      %arg0: tensor<2x32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_3_c_2_d_2, [{"d", ?}, {"a", "b", ?}]>})
func.func @non_minor_most_factor_non_divisible_same_open_dim_unchanged(
  %arg0: tensor<2x32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_3_c_2_d_2, [{?}, {"a", "b", ?}]>})
  -> tensor<2x8x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_3_c_2_d_2, [{"d"}, {"a", ?}, {"c"}]>]>}
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_3_c_2_d_2, [{"d"}, {?}, {"c"}]>]>} : (tensor<2x32xf32>) -> tensor<2x8x4xf32>
  return %0 : tensor<2x8x4xf32>
}

// CHECK-LABEL: func @merge_dim_minor_most_factor_non_divisible
func.func @merge_dim_minor_most_factor_non_divisible(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_3, [{"a"}, {"b"}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_3, [{"a", "b", ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @merge_dim_non_minor_most_factor_non_divisible
func.func @merge_dim_non_minor_most_factor_non_divisible(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3, [{"a"}, {}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0
  // CHECK-NOT:  sdy.sharding
  %0 = stablehlo.reshape %arg0 : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// Note that each axis on its own divides the size of the dimension, but
// together they don't, so only the major-most axis is propagated.
// CHECK-LABEL: func @merge_dim_non_minor_most_factor_non_divisible_multiple_axes
func.func @merge_dim_non_minor_most_factor_non_divisible_multiple_axes(
    %arg0: tensor<6x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3_b_3, [{"a", "b"}, {}]>}) -> tensor<24xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_3_b_3, [{"a", ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<6x4xf32>) -> tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @custom_call_custom_rule
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>})
// CHECK-SAME:      -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>}) {
func.func @custom_call_custom_rule(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>}) -> tensor<8xf32> {
  // CHECK:      stablehlo.custom_call @foo(%arg0) {
  // CHECK-SAME:     sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}]>]>,
  // CHECK-SAME:     sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i]) {i=8}, custom>
  // CHECK-SAME: } : (tensor<8xf32>) -> tensor<8xf32>
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i]) {i=8}, custom>} : (tensor<8xf32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @direct_arg_return_used_axes(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}, {?}]>}) {
func.func @direct_arg_return_used_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}, {?}]>}) {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @direct_arg_return_prefix_axes(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "c", ?}, {?}]>}) {
func.func @direct_arg_return_prefix_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "c", ?}, {?}]>}) {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @direct_arg_return_both_updated(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>}) {
func.func @direct_arg_return_both_updated(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b", ?}]>}) {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @direct_arg_return_sharding_on_arg(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>}) {
func.func @direct_arg_return_sharding_on_arg(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @direct_arg_return_sharding_on_result(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
func.func @direct_arg_return_sharding_on_result(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @func_out_sharding(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b", ?}]>})
// CHECK-SAME:  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
func.func @func_out_sharding(%arg0: tensor<8x8xf32>, %arg1: tensor<8x16xf32>)
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @token_func_output_skipped(
// CHECK-SAME:      %arg0: !stablehlo.token,
// CHECK-SAME:      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:  -> (!stablehlo.token, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
func.func @token_func_output_skipped(%arg0: !stablehlo.token, %arg1: tensor<8x16xf32>)
    -> (!stablehlo.token, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  %0 = stablehlo.add %arg1, %arg1 : tensor<8x16xf32>
  return %arg0, %0 : !stablehlo.token, tensor<8x16xf32>
}

// CHECK-LABEL: func @dynamic_shaped_func_output_skipped(
// CHECK-SAME:      %arg0: tensor<?x?xf32>,
// CHECK-SAME:      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:  -> (tensor<?x?xf32>, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
func.func @dynamic_shaped_func_output_skipped(%arg0: tensor<?x?xf32>, %arg1: tensor<8x16xf32>)
    -> (tensor<?x?xf32>, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  %0 = stablehlo.add %arg1, %arg1 : tensor<8x16xf32>
  return %arg0, %0 : tensor<?x?xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @func_result_intermediate_op_both_updated(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>}) {
func.func @func_result_intermediate_op_both_updated(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b", ?}]>}) {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @multiple_func_results(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>},
// CHECK-SAME:      tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b", ?}]>}) {
func.func @multiple_func_results(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>}, %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b", ?}]>}) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  return %arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @source_result_two_meshes_not_propagated(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3, [{"a", ?}, {?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3_another, [{?}, {?}]>}) {
// CHECK-NEXT: stablehlo.tanh %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_3, [{"a", ?}, {?}]>]>}
func.func @source_result_two_meshes_not_propagated(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3, [{"a", ?}, {?}]>})
   -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3_another, [{?}, {?}]>}) {
  %0 = stablehlo.tanh %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: func @operands_two_meshes_not_propagated(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3, [{"a"}, {?}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3_another, [{"a"}, {?}]>})
// CHECK-SAME:  -> tensor<8x8xf32> {
// CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
func.func @operands_two_meshes_not_propagated(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3, [{"a"}, {?}]>},
   %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_3_another, [{"a"}, {?}]>}) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
