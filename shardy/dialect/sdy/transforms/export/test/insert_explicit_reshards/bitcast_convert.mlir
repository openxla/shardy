// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @bitcast_convert_upcast
func.func @bitcast_convert_upcast(%arg0: tensor<4x2x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<4x2xui64> {
  // CHECK: %[[BITCAST_CONVERT:.*]] = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<4x2x2xui32>) -> tensor<4x2xui64>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[BITCAST_CONVERT]] <@mesh, [{}, {}]> : tensor<4x2xui64>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x2xui64>
  %0 = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x2x2xui32>) -> tensor<4x2xui64>
  return %0 :  tensor<4x2xui64>
}

// CHECK-LABEL: func @bitcast_convert_upcast_casting_dim_is_sharded
func.func @bitcast_convert_upcast_casting_dim_is_sharded(%arg0: tensor<4x2x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}) -> tensor<4x2xui64> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}]> : tensor<4x2x2xui32>
  // CHECK-NEXT: %[[BITCAST_CONVERT:.*]] = stablehlo.bitcast_convert %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<4x2x2xui32>) -> tensor<4x2xui64>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[BITCAST_CONVERT]] <@mesh, [{}, {}]> : tensor<4x2xui64>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x2xui64>
  // TODO(enver): Instead reshard only once.
  %0 = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x2x2xui32>) -> tensor<4x2xui64>
  return %0 :  tensor<4x2xui64>
}

// CHECK-LABEL: func @bitcast_convert_downcast
func.func @bitcast_convert_downcast(%arg0: tensor<4x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x2x2xui32> {
  // CHECK: %[[BITCAST_CONVERT:.*]] = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x2xui64>) -> tensor<4x2x2xui32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[BITCAST_CONVERT]] <@mesh, [{}, {}, {}]> : tensor<4x2x2xui32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x2x2xui32>
  %0 = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>}: (tensor<4x2xui64>) -> tensor<4x2x2xui32>
  return %0 :  tensor<4x2x2xui32>
}

// CHECK-LABEL: func @bitcast_convert_downcast_casting_dim_is_sharded
func.func @bitcast_convert_downcast_casting_dim_is_sharded(%arg0: tensor<4x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x2x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}){
  // CHECK: %[[BITCAST_CONVERT:.*]] = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x2xui64>) -> tensor<4x2x2xui32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[BITCAST_CONVERT]] <@mesh, [{"x"}, {}, {"y"}]> : tensor<4x2x2xui32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x2x2xui32>
  %0 = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>]>} : (tensor<4x2xui64>) -> tensor<4x2x2xui32>
  return %0 :  tensor<4x2x2xui32>
}
