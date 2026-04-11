// RUN: sdy_opt %s -split-input-file -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @named_computation
func.func @named_computation(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: sdy.named_computation<"foo">(%[[RESHARD]])
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD]] : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// -----
sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @named_computation_empty_block
func.func @named_computation_empty_block(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: sdy.named_computation<"foo">(%arg0)
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<210xf32>) {
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD]] : tensor<210xf32>
    sdy.return %arg1 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @named_computation_with_shardings
func.func @named_computation_with_shardings(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> tensor<12x2xi32> {
  %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, [{}, {}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %1 = stablehlo.concatenate %0#0, %0#1, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> tensor<12x2xi32>
  return %1 : tensor<12x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
  -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>} : tensor<8x2xi32>
  return %0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}
