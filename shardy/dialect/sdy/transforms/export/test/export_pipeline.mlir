// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-export-pipeline | FileCheck %s

// NOTE: We apply `sdy-add-data-flow-edges` first, to make sure
// `sdy-sink-data-flow-edges` is applied before any pass that operated on
// `ShardableDataFlowOpInterface` rather than `DataFlowEdgeOp`.

sdy.mesh @mesh3d = <["a"=4, "b"=4, "c"=4]>

// CHECK-LABEL: func @manual_computation_free_axes_non_divisible
func.func @manual_computation_free_axes_non_divisible(
    %arg0: tensor<4xf32>, %arg1: tensor<12xf32>, %arg2: tensor<24xf32>,
    %arg3: tensor<48xf32>, %arg4: tensor<96xf32>, %arg5: tensor<192xf32>)
    -> (tensor<4xf32>, tensor<12xf32>, tensor<24xf32>,
        tensor<48xf32>, tensor<96xf32>, tensor<192xf32>) {
  // CHECK-NEXT: sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
  // CHECK-SAME:   in_shardings=[<@mesh3d, [{"a"}]>, <@mesh3d, [{"a"}]>,
  // CHECK-SAME:                 <@mesh3d, [{"a", "b":(1)2}]>, <@mesh3d, [{"a", "b"}]>,
  // CHECK-SAME:                 <@mesh3d, [{"a", "b", "c":(1)2}]>, <@mesh3d, [{"a", "b", "c"}]>]
  // CHECK-SAME:   out_shardings=[<@mesh3d, [{"a"}]>, <@mesh3d, [{"a"}]>,
  // CHECK-SAME:                  <@mesh3d, [{"a", "b":(1)2}]>, <@mesh3d, [{"a", "b"}]>,
  // CHECK-SAME:                  <@mesh3d, [{"a", "b", "c":(1)2}]>, <@mesh3d, [{"a", "b", "c"}]>]
  // CHECK-SAME:   manual_axes={"a"}
  %0:6 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
    in_shardings=[<@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>,
                  <@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>,
                  <@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>]
    out_shardings=[<@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>,
                   <@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>,
                   <@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>]
    manual_axes={"a"} (%arg6: tensor<1xf32>, %arg7: tensor<3xf32>, %arg8: tensor<6xf32>, %arg9: tensor<12xf32>, %arg10: tensor<24xf32>, %arg11: tensor<48xf32>) {
    sdy.return %arg6, %arg7, %arg8, %arg9, %arg10, %arg11 : tensor<1xf32>, tensor<3xf32>, tensor<6xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>
  } : (tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>) -> (tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>)
  return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>
}

// CHECK-LABEL: func @all_reduce_of_replicated_to_unreduced
func.func @all_reduce_of_replicated_to_unreduced(
      %arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"c"}, {}], unreduced={"b"}>})
      -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"c"}, {}], unreduced={"b"}>}) {
  // CHECK-NEXT: %0 = sdy.replicated_to_unreduced
  // CHECK-NEXT: return %arg0 : tensor<16x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh3d, [{"c"}, {}], unreduced={"a", "b"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @update_input_output_shardings
// CHECK-SAME:    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{}, {"b"}]>}
// CHECK-SAME:    -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{}, {"b"}]>})
func.func @update_input_output_shardings(
    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"a", ?}, {"b"}]>})
    ->    (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"a":(1)2, ?}, {"b", "c"}]>}) {
  return %arg0 : tensor<2x4xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @named_computation_with_shardings
func.func @named_computation_with_shardings(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> tensor<12x2xi32> {
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT: %1:2 = call @foo(%0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %2 = sdy.reshard %1#0 <@mesh, [{}, {"a"}]> : tensor<8x2xi32>
  // CHECK-NEXT: %3 = sdy.reshard %1#1 <@mesh, [{}, {"a"}]> : tensor<4x2xi32>
  // CHECK-NEXT: %4 = stablehlo.concatenate %2, %3, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>}
  // CHECK-NEXT: return %4 : tensor<12x2xi32>
  %0:2 = sdy.named_computation<"foo">(%arg0, %arg1) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>] (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>)  {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %1 = stablehlo.concatenate %0#0, %0#1, dim=0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> tensor<12x2xi32>
  return %1 : tensor<12x2xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
// CHECK-SAME:       -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
// CHECK-SAME:       attributes {sdy.original_func_name = "foo"} {
// CHECK-NEXT:    return %arg0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
// CHECK-NEXT:  }
