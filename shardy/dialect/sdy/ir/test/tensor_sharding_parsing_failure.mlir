// RUN: sdy_opt %s -split-input-file -verify-diagnostics

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @priority_without_numbers(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+3 {{expecting priority in format 'p<number>', got: phigh}}
  // expected-error@+2 {{failed to parse Sdy_TensorSharding parameter 'dim_shardings' which is to be a `::llvm::ArrayRef<DimensionShardingAttr>`}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {"b"}phigh]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>

}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @priority_integer_overflow(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+3 {{expecting integer priority, got: p9999999999999999999}}
  // expected-error@+2 {{failed to parse Sdy_TensorSharding parameter 'dim_shardings' which is to be a `::llvm::ArrayRef<DimensionShardingAttr>`}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{"a"}, {"b"}p9999999999999999999]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

// expected-error@+2{{priorities with leading zeros are not allowed, got: p01}}
// expected-error@+1{{failed to parse Sdy_TensorSharding parameter 'dim_shardings' which is to be a `::llvm::ArrayRef<DimensionShardingAttr>`}}
func.func @priority_with_leading_zeros(%arg0 : tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@foo, [{"a"}p01, {"b"}]>}, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
