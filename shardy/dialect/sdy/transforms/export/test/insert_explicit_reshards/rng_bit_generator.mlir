// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @rng_bit_generator
func.func @rng_bit_generator(%arg0: tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> tensor<2xui64> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}]> : tensor<2xui64>
  // CHECK-NEXT: %output_state, %output = stablehlo.rng_bit_generator %[[RESHARD1]], algorithm =  DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>, <@mesh, [{"y"}, {"x":(2)2}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %output_state <@mesh, [{"x":(1)2}]> : tensor<2xui64>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD2]]
  %0, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}]>, <@mesh, [{"y"}, {"x":(2)2}]>]>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4x1000xui32>)
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}]>]>} : tensor<2xui64>
  return %1 : tensor<2xui64>
}
