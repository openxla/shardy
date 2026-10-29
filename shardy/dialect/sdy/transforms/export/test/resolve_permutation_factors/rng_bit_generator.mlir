// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s --check-prefixes=CHECK,HALO

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @rng_bit_generator_ui64_state
// CHECK-SAME: (%[[STATE:.*]]: tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
// CHECK-SAME: -> (tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
// CHECK-SAME:     tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
func.func @rng_bit_generator_ui64_state(
    %arg0: tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}
) -> (
    tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
    tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}
) {
  // REPL-NEXT: %[[OUT_STATE:.*]], %[[OUT_REPL:.*]] = stablehlo.rng_bit_generator %[[STATE]], algorithm = DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>, <@mesh, [{}, {}]>]>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<8x16xf32>)
  // REPL-NEXT: %[[OUT:.*]] = sdy.reshard %[[OUT_REPL]] <@mesh, [{"a"}, {"b"}]> : tensor<8x16xf32>

  // HALO-NEXT: %[[OUT_STATE:.*]], %[[OUT:.*]] = stablehlo.rng_bit_generator %[[STATE]], algorithm = DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>, <@mesh, [{"a"}, {"b"}]>]>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<8x16xf32>)
  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = DEFAULT {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>, <@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<2xui64>) -> (tensor<2xui64>, tensor<8x16xf32>)
  // CHECK-NEXT: return %[[OUT_STATE]], %[[OUT]] : tensor<2xui64>, tensor<8x16xf32>
  return %output_state, %output : tensor<2xui64>, tensor<8x16xf32>
}

// CHECK-LABEL: func @rng_bit_generator_non_32_64_bit_state_always_replicated
// CHECK-SAME: (%[[STATE:.*]]: tensor<2xui16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
// CHECK-SAME: -> (tensor<2xui16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
// CHECK-SAME:     tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
func.func @rng_bit_generator_non_32_64_bit_state_always_replicated(
    %arg0: tensor<2xui16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}
) -> (
    tensor<2xui16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
    tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}
) {
  // CHECK-NEXT: %[[OUT_STATE:.*]], %[[OUT_REPL:.*]] = stablehlo.rng_bit_generator %[[STATE]], algorithm = DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>, <@mesh, [{}, {}]>]>} : (tensor<2xui16>) -> (tensor<2xui16>, tensor<8x16xf32>)
  // CHECK-NEXT: %[[OUT:.*]] = sdy.reshard %[[OUT_REPL]] <@mesh, [{"a"}, {"b"}]> : tensor<8x16xf32>
  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = DEFAULT {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>, <@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<2xui16>) -> (tensor<2xui16>, tensor<8x16xf32>)
  // CHECK-NEXT: return %[[OUT_STATE]], %[[OUT]] : tensor<2xui16>, tensor<8x16xf32>
  return %output_state, %output : tensor<2xui16>, tensor<8x16xf32>
}
