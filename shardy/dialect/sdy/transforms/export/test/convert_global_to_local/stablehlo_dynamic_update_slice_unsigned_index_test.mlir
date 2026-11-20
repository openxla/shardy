// RUN: sdy_opt %s -sdy-convert-global-to-local -verify-diagnostics

sdy.mesh @mesh_4 = <["x"=4]>

// Unsigned ui8 constant 190 crosses shard boundary 192 on a 4 shard dimension
// of size 256 and must fail legalization when not replicated upstream.
func.func @cross_shard_unsigned_msb_constant_index_not_replicated(
    %arg0: tensor<256xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>},
    %arg1: tensor<4xf32>) -> (tensor<256xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>}) {
  %c190 = stablehlo.constant dense<190> : tensor<ui8>
  // expected-error @+2 {{partitioned dynamic-update-slice crossing shard boundaries requires upstream resharding/replication}}
  // expected-error @+1 {{failed to legalize operation 'stablehlo.dynamic_update_slice' that was explicitly marked illegal}}
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c190 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>} : (tensor<256xf32>, tensor<4xf32>, tensor<ui8>) -> tensor<256xf32>
  return %0 : tensor<256xf32>
}
