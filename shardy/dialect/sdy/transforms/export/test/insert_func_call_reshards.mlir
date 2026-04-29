// RUN: sdy_opt %s -split-input-file -sdy-insert-func-call-reshards | FileCheck %s

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func @main
// CHECK-SAME: (%arg0: tensor<8xf32>) -> tensor<8xf32> {
// CHECK-NEXT:  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
// CHECK-NEXT:}
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME: (%arg0: tensor<8xf32>) -> tensor<8xf32> {
// CHECK-NEXT:  return %arg0 : tensor<8xf32>
// CHECK-NEXT:}
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  return %arg0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
// CHECK-NEXT:  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  return %1 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func @multiple_same_calls_different_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @non_flat_nested_calls_mixed_shardings
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %2 = call @bar_0(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %3 = call @bar_1(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %3 : tensor<8xf32>
// CHECK-NEXT:}
func.func @non_flat_nested_calls_mixed_shardings(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %2 = call @bar_0(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %3 = call @bar_1(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
}
// CHECK-LABEL: func private @bar
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
// CHECK-NEXT:}
func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: func private @foo
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %1 = call @bar_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
// CHECK-NEXT:}
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %1 = call @bar_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
// CHECK-LABEL: func private @bar_0
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
// CHECK-NEXT:}
func.func private @bar_0(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: func private @bar_1
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
// CHECK-NEXT:}
func.func private @bar_1(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @calls_same_funcs_with_manual_axes_one_without
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
// CHECK-NEXT:  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
// CHECK-NEXT:    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %4 = func.call @foo(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    sdy.return %4 : tensor<4xf32>
// CHECK-NEXT:  } : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = sdy.reshard %0 <@mesh, [{"y"}]> : tensor<8xf32>
// CHECK-NEXT:  %2 = call @bar(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %2 : tensor<8xf32>
// CHECK-NEXT:}
func.func @calls_same_funcs_with_manual_axes_one_without(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
    %3 = func.call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4xf32>) -> tensor<4xf32>
    %4 = func.call @foo(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4xf32>) -> tensor<4xf32>
    sdy.return %4 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = sdy.reshard %0 <@mesh, [{"y"}]> : tensor<8xf32>
  %2 = call @bar(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %2 : tensor<8xf32>
}
// CHECK-LABEL: func private @foo
// CHECK-SAME: (%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:  return %0 : tensor<4xf32>
// CHECK-NEXT:}
func.func private @foo(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func private @foo_0
// CHECK-SAME: (%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
// CHECK-NEXT:  return %0 : tensor<4xf32>
// CHECK-NEXT:}
func.func private @foo_0(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func private @bar
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
// CHECK-NEXT:}
func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
// CHECK-NEXT:  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %2 = stablehlo.add %0, %1 : tensor<8x2xi32>
// CHECK-NEXT:  return %2 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func @multiple_same_calls_different_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @single_call_different_argument_and_func_input_sharding
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> tensor<8x2xi32> {
// CHECK-NEXT:  %0 = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x2xi32>
// CHECK-NEXT:  %1 = call @baz(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  return %1 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func @single_call_different_argument_and_func_input_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> tensor<8x2xi32> {
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x2xi32>
  %1 = call @baz(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @single_call_different_argument_and_func_input_sharding_argument_reused
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> tensor<8x2xi32> {
// CHECK-NEXT:  %0 = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x2xi32>
// CHECK-NEXT:  %1 = call @baz(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %2 = stablehlo.add %1, %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  return %2 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func @single_call_different_argument_and_func_input_sharding_argument_reused(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> tensor<8x2xi32> {
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x2xi32>
  %1 = call @baz(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.add %1, %arg0 : tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @single_call_func_result_empty_sharding_call_has_sharding
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
// CHECK-NEXT:  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %1 = sdy.reshard %0 <@mesh, [{}, {"y"}]> : tensor<8x2xi32>
// CHECK-NEXT:  return %1 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func @single_call_func_result_empty_sharding_call_has_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
// CHECK-NEXT:  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %2 = sdy.reshard %1 <@mesh, [{"x"}, {"y"}]> : tensor<8x2xi32>
// CHECK-NEXT:  return %2 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func @multiple_same_calls_different_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @non_flat_nested_calls_mixed_shardings
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %2 = call @bar_0(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %3 = call @bar_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  %4 = sdy.reshard %3 <@mesh, [{"y"}]> : tensor<8xf32>
// CHECK-NEXT:  return %4 : tensor<8xf32>
// CHECK-NEXT:}
func.func @non_flat_nested_calls_mixed_shardings(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %2 = call @bar_0(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  %3 = call @bar_0(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
}
// CHECK-LABEL: func private @bar
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
// CHECK-NEXT:}
func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: func private @foo
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  %1 = call @bar_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:  return %1 : tensor<8xf32>
// CHECK-NEXT:}
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %1 = call @bar_0(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
// CHECK-LABEL: func private @bar_0
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
// CHECK-NEXT:}
func.func private @bar_0(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: func private @bar_1
// CHECK-SAME: (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
// CHECK-NEXT:  return %0 : tensor<8xf32>
// CHECK-NEXT:}
func.func private @bar_1(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_different_shardings
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
// CHECK-NEXT:  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %2 = sdy.reshard %1 <@mesh, [{"x"}, {"y"}]> : tensor<8x2xi32>
// CHECK-NEXT:  %3 = stablehlo.add %0, %2 : tensor<8x2xi32>
// CHECK-NEXT:  return %3 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func @multiple_same_calls_different_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @multiple_same_calls_multiple_outputs_different_shardings
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
// CHECK-NEXT:  %0:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
// CHECK-NEXT:  %1 = stablehlo.divide %0#0, %0#1 : tensor<8x2xi32>
// CHECK-NEXT:  %2:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
// CHECK-NEXT:  %3 = sdy.reshard %2#1 <@mesh, [{"y"}, {"x"}]> : tensor<8x2xi32>
// CHECK-NEXT:  %4 = sdy.reshard %2#0 <@mesh, [{"x"}, {"y"}]> : tensor<8x2xi32>
// CHECK-NEXT:  %5 = stablehlo.divide %4, %3 : tensor<8x2xi32>
// CHECK-NEXT:  %6 = stablehlo.add %1, %5 : tensor<8x2xi32>
// CHECK-NEXT:  return %6 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func @multiple_same_calls_multiple_outputs_different_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %1 = stablehlo.divide %0#0, %0#1 : tensor<8x2xi32>
  %2:2 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %3 = stablehlo.divide %2#0, %2#1 : tensor<8x2xi32>
  %4 = stablehlo.add %1, %3 : tensor<8x2xi32>
  return %4 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0, %0 : tensor<8x2xi32>, tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0, %0 : tensor<8x2xi32>, tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0, %0 : tensor<8x2xi32>, tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
  return %0, %0 : tensor<8x2xi32>, tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @single_call_func_result_empty_sharding_call_has_sharding
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
// CHECK-NEXT:  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %1 = sdy.reshard %0 <@mesh, [{}, {"y"}]> : tensor<8x2xi32>
// CHECK-NEXT:  return %1 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func @single_call_func_result_empty_sharding_call_has_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2]>
// CHECK-LABEL: func @three_calls_same_origin_func_with_two_calls_results_no_out_sharding
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
// CHECK-NEXT:  %0 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %1 = sdy.reshard %0 <@mesh, [{}, {"y"}]> : tensor<8x2xi32>
// CHECK-NEXT:  %2 = call @baz_0(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %3 = call @baz_0(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
// CHECK-NEXT:  %4 = stablehlo.add %1, %2 : tensor<8x2xi32>
// CHECK-NEXT:  return %4 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func @three_calls_same_origin_func_with_two_calls_results_no_out_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = call @baz_0(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @baz_0(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %3 = stablehlo.add %0, %1 : tensor<8x2xi32>
  return %3 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
// CHECK-LABEL: func private @baz_0
// CHECK-SAME: (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
// CHECK-NEXT:  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:  return %0 : tensor<8x2xi32>
// CHECK-NEXT:}
func.func private @baz_0(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
