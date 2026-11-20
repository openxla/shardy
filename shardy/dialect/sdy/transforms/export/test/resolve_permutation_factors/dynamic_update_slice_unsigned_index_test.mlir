// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Unsigned ui8 constant 190 has its high bit set and crosses shard boundary 192
// on a 4 shard dimension of size 256. Factor resolution must replicate.
// CHECK-LABEL: func @dynamic_update_slice_cross_shard_ui8_msb
// CHECK:      %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<256x8xf32>
// CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[RESHARD0]], %arg1, %{{.*}}, %{{.*}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[DUS]] <@mesh, [{"a", "b"}, {}]> : tensor<256x8xf32>
// CHECK-NEXT: return %[[RESHARD1]]
func.func @dynamic_update_slice_cross_shard_ui8_msb(%arg0: tensor<256x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {}]>}, %arg1: tensor<4x8xf32>) -> (tensor<256x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {}]>}) {
  %c190 = sdy.constant dense<190> : tensor<ui8>
  %c0 = sdy.constant dense<0> : tensor<ui8>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c190, %c0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b"}, {}]>]>} : (tensor<256x8xf32>, tensor<4x8xf32>, tensor<ui8>, tensor<ui8>) -> tensor<256x8xf32>
  return %0 : tensor<256x8xf32>
}

// Unsigned ui64 constant UINT64_MAX clamps to 8 on an indivisible 10 element
// dimension (shard size 3 across 4 shards), crossing shard boundary 9.
// CHECK-LABEL: func @dynamic_update_slice_cross_shard_ui64_max
// CHECK:      %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<10x8xf32>
// CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[RESHARD0]], %arg1, %{{.*}}, %{{.*}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[DUS]] <@mesh, [{"a", "b"}, {}]> : tensor<10x8xf32>
// CHECK-NEXT: return %[[RESHARD1]]
func.func @dynamic_update_slice_cross_shard_ui64_max(%arg0: tensor<10x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {}]>}, %arg1: tensor<2x8xf32>) -> (tensor<10x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {}]>}) {
  %cmax = sdy.constant dense<18446744073709551615> : tensor<ui64>
  %c0 = sdy.constant dense<0> : tensor<ui64>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %cmax, %c0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b"}, {}]>]>} : (tensor<10x8xf32>, tensor<2x8xf32>, tensor<ui64>, tensor<ui64>) -> tensor<10x8xf32>
  return %0 : tensor<10x8xf32>
}
