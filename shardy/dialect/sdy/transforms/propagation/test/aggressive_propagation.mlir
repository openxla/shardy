// RUN: sdy_opt %s -sdy-aggressive-propagate="propagation-strategy=aggressive" -verify-diagnostics 2>&1 | FileCheck %s

sdy.mesh @empty_mesh = <[]>
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @no_conflict(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>},
// CHECK-SAME:      %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", ?}, {?}]>})
// CHECK-SAME:  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>}) {
func.func @no_conflict(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
                       %arg1: tensor<8x8xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: stablehlo.dot_general %[[ADD]], %arg2
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {?}]>]>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @fake_conflict_between_two_non_contracting_dims(
// CHECK-SAME:      %arg0: tensor<256x512xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>})
// CHECK-SAME:  -> (tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"a", ?}]>}) {
func.func @fake_conflict_between_two_non_contracting_dims(%arg0: tensor<256x512xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>},
                                                          %arg1: tensor<128x512xf32>)
          -> (tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"a", ?}]>}) {
  // CHECK-NEXT: %0 = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}, {"a", ?}]>]>}
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}, {"a", ?}]>]>} :
    (tensor<256x512xf32>, tensor<128x512xf32>) -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// CHECK-LABEL: func @fake_conflict_between_contracting_and_non_contracting_dims(
// CHECK-SAME:      %arg0: tensor<256x512xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}, {"b", "c"}]>},
// CHECK-SAME:      %arg1: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c", "b"}, {"a"}]>})
// CHECK-SAME:  -> (tensor<256x128xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", "b", ?}]>}) {
func.func @fake_conflict_between_contracting_and_non_contracting_dims(%arg0: tensor<256x512xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}, {"b", "c"}]>},
                                                                      %arg1: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c", "b"}, {"a"}]>}) -> tensor<256x128xf32> {
  // CHECK-NEXT: %0 = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", ?}, {"c", "b", ?}]>]>}
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1] :
    (tensor<256x512xf32>, tensor<128x512xf32>) -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// CHECK-LABEL: func @fake_conflict_closed_dims(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{}, {"b", "c", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"b", "c", ?}]>}) {
func.func @fake_conflict_closed_dims(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"b"}]>},
                                     %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{}, {"b", "c", ?}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", ?}, {"b", "c", ?}]>]>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @real_conflict_across_factors(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}, {?}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{?}, {"a"}]>})
// CHECK-SAME:  -> tensor<8x8xf32> {
func.func @real_conflict_across_factors(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}, {?}]>},
                                        %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{?}, {"a"}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1
  // CHECK-NOT:    sdy.sharding
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @real_conflict_within_a_factor(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", ?}, {}]>})
// CHECK-SAME:  -> tensor<8x8xf32> {
func.func @real_conflict_within_a_factor(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {}]>},
                                         %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", ?}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1
  // CHECK-NOT:    sdy.sharding
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @real_and_fake_conflicts(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", ?}, {"a", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{?}, {"a", ?}]>}) {
func.func @real_and_fake_conflicts(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {?}]>},
                                   %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", ?}, {"a", ?}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{?}, {"a", ?}]>]>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @empty_mesh_replaced_closed_dim_respected(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {}]>})
func.func @empty_mesh_replaced_closed_dim_respected(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{?}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
