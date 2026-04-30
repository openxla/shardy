// RUN: sdy_opt %s -split-input-file -sdy-flatten-call-graph | FileCheck %s

// CHECK-LABEL: func @singleton(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @singleton(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: func @simple_call_graph(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @simple_call_graph(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  return %1 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: func @main_calls_foo_twice(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @foo_0(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @main_calls_foo_twice(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  %1 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge1, %edge2 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "foo"} {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return


// -----

// CHECK-LABEL: func @main_calls_foo_calls_bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @main_calls_foo_calls_bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  return %1 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  %1 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %2 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: func @main_calls_foo_calls_bar_twice(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @main_calls_foo_calls_bar_twice(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  return %1 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar_0(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    stablehlo.add
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  %1 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  %2 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %2 : tensor<8xi32>
  %3 = stablehlo.add %edge1, %edge2 : tensor<8xi32>
  return %3 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "bar"} {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return

// -----

// CHECK-LABEL: func @simple_non_flat(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar_0(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @simple_non_flat(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  %1 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge1, %edge2 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  %1 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge : tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "bar"} {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return

// -----

// CHECK-LABEL: func @main_calls_foo_twice_and_foo_calls_bar_twice(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @foo_3(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @main_calls_foo_twice_and_foo_calls_bar_twice(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  %1 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge1, %edge2 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar_0(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    stablehlo.add
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  %1 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  %2 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %2 : tensor<8xi32>
  %3 = stablehlo.add %edge1, %edge2 : tensor<8xi32>
  return %3 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "bar"} {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return

// CHECK-LABEL: func private @bar_1(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "bar"} {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return

// CHECK-LABEL: func private @bar_0_2(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "bar"} {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return

// CHECK-LABEL: func private @foo_3(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "foo"} {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar_1(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar_0_2(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    stablehlo.add

// -----

// CHECK-LABEL: func @main_calls_foo_twice_and_foo_calls_bar_once(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @foo_1(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @main_calls_foo_twice_and_foo_calls_bar_once(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  %1 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge1, %edge2 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  %1 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge: tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "bar"} {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return

// CHECK-LABEL: func private @foo_1(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "foo"} {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar_0(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return

// -----

// CHECK-LABEL:func.func @complex_non_flat(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:   %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
// CHECK-NEXT:   %2 = call @bar_4(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %3 = sdy.func_data_flow_edge %2 : tensor<8xi32>
// CHECK-NEXT:   return %1, %3 : tensor<8xi32>, tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:   %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
// CHECK-NEXT:   %1 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
// CHECK-NEXT:   %3 = call @bar_1(%0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %4 = sdy.func_data_flow_edge %3 : tensor<8xi32>
// CHECK-NEXT:   %5 = stablehlo.add %2, %4 : tensor<8xi32>
// CHECK-NEXT:   %6 = call @baz_2(%5) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %7 = sdy.func_data_flow_edge %6 : tensor<8xi32>
// CHECK-NEXT:   return %7 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:   %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
// CHECK-NEXT:   %1 = call @baz(%0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
// CHECK-NEXT:   return %2 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @baz(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:   %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
// CHECK-NEXT:   return %0 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @baz_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME: attributes {sdy.original_func_name = "baz"} {
// CHECK-NEXT:   %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
// CHECK-NEXT:   return %0 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @bar_1(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME: attributes {sdy.original_func_name = "bar"} {
// CHECK-NEXT:   %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
// CHECK-NEXT:   %1 = call @baz_0(%0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
// CHECK-NEXT:   return %2 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @baz_2(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME: attributes {sdy.original_func_name = "baz"} {
// CHECK-NEXT:   %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
// CHECK-NEXT:   return %0 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @baz_3(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME: attributes {sdy.original_func_name = "baz"} {
// CHECK-NEXT:   %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
// CHECK-NEXT:   return %0 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @bar_4(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME: attributes {sdy.original_func_name = "bar"} {
// CHECK-NEXT:   %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
// CHECK-NEXT:   %1 = call @baz_3(%0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
// CHECK-NEXT:   return %2 : tensor<8xi32>
// CHECK-NEXT: }
func.func @complex_non_flat(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  %1 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge1, %edge2 : tensor<8xi32>, tensor<8xi32>
}
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  %1 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  %2 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %2 : tensor<8xi32>
  %3 = stablehlo.add %edge1, %edge2 : tensor<8xi32>
  %4 = call @baz(%3) : (tensor<8xi32>) -> tensor<8xi32>
  %edge3 = sdy.func_data_flow_edge %4 : tensor<8xi32>
  return %edge3 : tensor<8xi32>
}

func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  %1 = call @baz(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge: tensor<8xi32>
}

func.func private @baz(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["a"=2, "b"=2]>

// CHECK-LABEL: func @simple_non_flat_sharding_on_func_arguments(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(%arg0)
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @foo_0(%arg0)
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @simple_non_flat_sharding_on_func_arguments(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  %1 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge1, %edge2 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo_0(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "foo"} {
// CHECK-NEXT:    sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
// CHECK-NEXT:    return

// -----

sdy.mesh @mesh = #sdy.mesh<["a"=2, "b"=2]>

// CHECK-LABEL: func @simple_non_flat_sharding_on_func_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(%arg0)
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @foo_0(%arg0)
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @simple_non_flat_sharding_on_func_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  %1 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge1, %edge2 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
// CHECK-NEXT:    sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  %0 = sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo_0(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
// CHECK-SAME:  attributes {sdy.original_func_name = "foo"} {
// CHECK-NEXT:    sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
// CHECK-NEXT:    return

// -----

sdy.mesh @mesh = #sdy.mesh<["a"=2, "b"=2]>

// CHECK-LABEL: func @simple_non_flat_non_matching_sharding_on_func_results_and_second_call_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @foo_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:    sdy.func_data_flow_edge {{.*}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>}
// CHECK-NEXT:    return
func.func @simple_non_flat_non_matching_sharding_on_func_results_and_second_call_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  %1 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : tensor<8xi32>
  return %edge1, %edge2 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
// CHECK-NEXT:    sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  %0 = sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo_0(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
// CHECK-SAME:  attributes {sdy.original_func_name = "foo"} {
// CHECK-NEXT:    sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
// CHECK-NEXT:    return

// -----

sdy.mesh @mesh = #sdy.mesh<["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @simple_non_flat_non_matching_sharding_on_func_results_and_both_call_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @foo_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func @simple_non_flat_non_matching_sharding_on_func_results_and_both_call_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : tensor<8xi32>
  %1 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
  %edge2 = sdy.func_data_flow_edge %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}]>]>} : tensor<8xi32>
  return %edge1, %edge2 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
// CHECK-NEXT:    sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  %0 = sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo_0(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
// CHECK-SAME:  attributes {sdy.original_func_name = "foo"} {
// CHECK-NEXT:    sdy.func_data_flow_edge %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>}
// CHECK-NEXT:    return

// -----

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func @simple_non_flat_with_manual_computations(
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    sdy.manual_computation
// CHECK-NEXT:      stablehlo.concatenate
// CHECK-NEXT:      func.call @bar_0
// CHECK-NEXT:      sdy.func_data_flow_edge
// CHECK-NEXT:      sdy.return
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func @simple_non_flat_with_manual_computations(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge1 = sdy.func_data_flow_edge %0 : tensor<8xi32>
  %1 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a"}]>] out_shardings=[<@mesh, [{"a"}]>] manual_axes={"a"} (%arg2: tensor<4xi32>) {
    %2 = stablehlo.concatenate %arg2, %arg2, dim=0 : (tensor<4xi32>, tensor<4xi32>) -> tensor<8xi32>
    %3 = func.call @bar(%2) : (tensor<8xi32>) -> tensor<8xi32>
    %edge3 = sdy.func_data_flow_edge %3 : tensor<8xi32>
    sdy.return %arg2 : tensor<4xi32>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %edge1, %1 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  %1 = call @bar(%0) : (tensor<8xi32>) -> tensor<8xi32>
  %edge = sdy.func_data_flow_edge %1 : tensor<8xi32>
  return %edge : tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "bar"} {
// CHECK-NEXT:    sdy.func_data_flow_edge
// CHECK-NEXT:    return

