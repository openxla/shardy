// RUN: sdy_opt %s -sdy-insert-explicit-reshards='avoid-reshards-on-named-computations=true' -sdy-insert-explicit-reshards='avoid-reshards-on-named-computations=true' | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=4]>

//===----------------------------------------------------------------------===//
// Named computations tests
// More tests are in insert_explicit_reshards/data_flow_ops.mlir
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @named_computation
func.func @named_computation(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}]>}) {
  // CHECK-NEXT: sdy.named_computation<"foo">(%arg0)
  // CHECK-SAME: in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"z"}]>
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"z"}]>] (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"z"}]> : tensor<210xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD]] : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @one_argument_to_multiple_named_computations(
func.func @one_argument_to_multiple_named_computations(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}]>}) {
  // CHECK-NEXT: %[[NC0:.*]] = sdy.named_computation<"foo">(%arg0)
  // CHECK-SAME: in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  // CHECK: %[[NC1:.*]] = sdy.named_computation<"foo">(%arg0)
  // CHECK-SAME: in_shardings=[<@mesh, [{"z"}]>] out_shardings=[<@mesh, [{"z"}]>
  %1 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"z"}]>] out_shardings=[<@mesh, [{"z"}]>] (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}]>]>} : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[NC0]], %[[NC1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}]>]>}
  // CHECK-NEXT: return %[[ADD]]
  %3 = stablehlo.add %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}]>]>} : tensor<210xf32>
  return %3 : tensor<210xf32>
}

// CHECK-LABEL: func @different_arguments_to_multiple_named_computations_with_same_input_output_shardings
func.func @different_arguments_to_multiple_named_computations_with_same_input_output_shardings(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NEXT: %[[NC0:.*]] = sdy.named_computation<"foo">(%arg0)
  // CHECK-SAME: in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<210xf32>) {
    %3 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
    sdy.return %3 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>}
  // CHECK-NEXT: %[[NC1:.*]] = sdy.named_computation<"foo">(%[[NEGATE]])
  // CHECK-SAME: in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>
  %1 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  %2 = sdy.named_computation<"foo">(%1) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<210xf32>) {
    %3 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
    sdy.return %3 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %4 = stablehlo.add %0, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %4 : tensor<210xf32>
}
