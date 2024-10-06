// RUN: sdy_opt %s -split-input-file -verify-diagnostics

func.func @invalid_operand_type(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{expected the type of the 0'th block argument to match the type of the corresponding operand: 'tensor<4x2xi32>' vs 'tensor<8x2xi32>'}}
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<4x2xi32>) {
    %1 = stablehlo.custom_call @foo(%arg1) : (tensor<4x2xi32>) -> tensor<8x2xi32>
    sdy.return %1 : tensor<8x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

func.func @operand_count_mismatch(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{number of block arguments must match the number of operands: 2 != 1}}
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
    sdy.return %arg1 : tensor<8x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

func.func @invalid_result_type(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{expected the type of the 0'th returned value to match the type of the corresponding result: 'tensor<4x2xi32>' vs 'tensor<8x2xi32>'}}
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>) {
    %1 = stablehlo.custom_call @foo(%arg1) : (tensor<8x2xi32>) -> tensor<4x2xi32>
    sdy.return %1 : tensor<4x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

func.func @result_count_mismatch(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{number of returned values must match the number of results: 2 != 1}}
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>) {
    sdy.return %arg1, %arg1 : tensor<8x2xi32>, tensor<8x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @result_shardings_mismatch(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{op out_shardings shardings don't match number of values: 2 shardings vs 1 values}}
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
    sdy.return %arg1 : tensor<8x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @operand_shardings_mismatch(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{op in_shardings shardings don't match number of values: 2 shardings vs 1 values}}
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"b"}, {}]>, <@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
    sdy.return %arg1 : tensor<8x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @sharding_rank_mismatch(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{op in_shardings - sharding doesn't match tensor rank: 3 != 2}}
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"b"}, {}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
    sdy.return %arg1 : tensor<8x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

func.func @unknown_mesh(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // expected-error@+1 {{op in_shardings - unknown mesh: @unknown_mesh}}
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@unknown_mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] (%arg1: tensor<8x2xi32>) {
    sdy.return %arg1 : tensor<8x2xi32>
  }: (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
