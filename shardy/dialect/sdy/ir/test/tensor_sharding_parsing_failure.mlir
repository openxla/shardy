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

// expected-error@+2 {{priorities with leading zeros are not allowed, got: p01}}
// expected-error@+1 {{failed to parse Sdy_TensorSharding parameter 'dim_shardings' which is to be a `::llvm::ArrayRef<DimensionShardingAttr>`}}
func.func @priority_with_leading_zeros(%arg0 : tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@foo, [{"a"}p01, {"b"}]>}, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @missing_equals_after_replicated(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+3 {{expected '='}}
  // expected-error@+2 {{failed to parse Sdy_TensorSharding parameter 'replicated_axes'}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated{"a", "b"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @missing_l_brace_after_unreduced(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+3 {{expected '{'}}
  // expected-error@+2 {{failed to parse Sdy_TensorSharding parameter 'unreduced_axes'}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced=["a", "b"]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @missing_r_brace_after_unreduced(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+3 {{expected '}'}}
  // expected-error@+2 {{failed to parse Sdy_TensorSharding parameter 'unreduced_axes'}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced={"a", "b"[]}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @invalid_replicated_axis_list(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+5 {{expected string}}
  // expected-error@+4 {{failed to parse Sdy_AxisRef parameter 'name' which is to be a `::llvm::StringRef`}}
  // expected-error@+3 {{failed to parse axis list which is expected to be an `ArrayRef<AxisRefAttr>`}}
  // expected-error@+2 {{failed to parse Sdy_TensorSharding parameter 'replicated_axes'}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a", b}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @invalid_unreduced_axis_list(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+5 {{expected string}}
  // expected-error@+4 {{failed to parse Sdy_AxisRef parameter 'name' which is to be a `::llvm::StringRef`}}
  // expected-error@+3 {{failed to parse axis list which is expected to be an `ArrayRef<AxisRefAttr>`}}
  // expected-error@+2 {{failed to parse Sdy_TensorSharding parameter 'unreduced_axes'}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unreduced={"a", b}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @invalid_replicated_then_unreduced_no_comma_in_between(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+3 {{expected '='}}
  // expected-error@+2 {{failed to parse Sdy_TensorSharding parameter 'replicated_axes'}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated unreduced={"a", "b"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @comma_without_replicated_or_unreduced(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+2 {{failed to parse Sdy_TensorSharding, expected valid named axis list after comma}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}],>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @unknown_keyword_after_comma(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+2 {{failed to parse Sdy_TensorSharding, expected valid named axis list after comma}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], unknown={"a"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @replicated_then_trailing_comma(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+2 {{failed to parse Sdy_TensorSharding, expected valid named axis list after comma}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, >]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @replicated_and_unreduced_then_trailing_comma(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+2 {{failed to parse Sdy_TensorSharding, expected valid named axis list after comma}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, unreduced={"b"}, >]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @duplicate_replicated_axes(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+2 {{failed to parse Sdy_TensorSharding, expected valid named axis list after comma}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, replicated={"b"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @foo = <["a"=2,"b"=4]>

func.func @duplicate_unreduced_axes(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+2 {{failed to parse Sdy_TensorSharding, expected valid named axis list after comma}}
  // expected-error@+1 {{failed to parse Sdy_TensorShardingPerValue parameter 'shardings' which is to be a `::llvm::ArrayRef<TensorShardingAttr>`}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@foo, [{}, {}], replicated={"a"}, unreduced={"b"}, unreduced={"b"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
