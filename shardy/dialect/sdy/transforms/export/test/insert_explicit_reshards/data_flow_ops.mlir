// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @named_computation
func.func @named_computation(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: sdy.named_computation<"foo">(%[[RESHARD]])
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD]] : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @named_computation_empty_block
func.func @named_computation_empty_block(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: sdy.named_computation<"foo">(%arg0)
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<210xf32>) {
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD]] : tensor<210xf32>
    sdy.return %arg1 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @named_computation_different_argument_and_input_shardings
func.func @named_computation_different_argument_and_input_shardings(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}]> : tensor<210xf32>
  // CHECK-NEXT: sdy.named_computation<"foo">(%[[RESHARD]])
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @named_computation_different_result_and_output_shardings
func.func @named_computation_different_result_and_output_shardings(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0)
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %[[NC]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh, [{"y"}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<210xf32>
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @case
func.func @case(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}]>}, %arg1: tensor<i32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = "stablehlo.case"(%arg1) ({
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.abs %[[RESHARD]]
    %2 = stablehlo.abs %arg0 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]] : tensor<210xf32>
    stablehlo.return %2 : tensor<210xf32>
  }, {
    %2 = stablehlo.cosine %arg0 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %2 <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.abs %[[RESHARD]]
    %3 = stablehlo.abs %2 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
    stablehlo.return %3 : tensor<210xf32>
  }, {
    %2 = stablehlo.abs %arg0 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x":(1)2}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]] : tensor<210xf32>
    stablehlo.return %2 : tensor<210xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<i32>) -> tensor<210xf32>
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @case_argument_used_outside_block
func.func @case_argument_used_outside_block(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg1: tensor<i32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  %1 = "stablehlo.case"(%arg1) ({
    // CHECK: stablehlo.return %[[RESHARD1]]
    stablehlo.return %arg0 : tensor<210xf32>
  }, {
    // CHECK: %[[RESHARD2:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[RESHARD2]]
    // CHECK-NEXT: stablehlo.return %[[ABS]] : tensor<210xf32>
    %4 = stablehlo.abs %arg0 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    stablehlo.return %4 : tensor<210xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<i32>) -> tensor<210xf32>
  // CHECK: stablehlo.add %[[NEGATE]], %arg0
  %2 = stablehlo.add %0, %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  %3 = stablehlo.add %1, %2 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
  return %3 : tensor<210xf32>
}

// CHECK-LABEL: func @while
func.func @while(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<32> : tensor<i32>
  %c_2 = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}]> : tensor<210xf32>
  // CHECK-NEXT stablehlo.while(%iterArg = %[[RESHARD]], %iterArg_2 = %c)
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c) : tensor<210xf32>, tensor<i32> attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>, <@mesh, []>]>}
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.dot %iterArg, %iterArg : (tensor<210xf32>, tensor<210xf32>) -> tensor<f32>
    %4 = stablehlo.compare  LT, %3, %c_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %2, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %iterArg <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
    %3 = stablehlo.negate %iterArg {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    %4 = stablehlo.add %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}: tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]], %{{.*}} : tensor<210xf32>, tensor<i32>
    stablehlo.return %4, %2 : tensor<210xf32>, tensor<i32>
  }
  %1 = stablehlo.negate %0#0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @while_argument_used_outside_block
func.func @while_argument_used_outside_block(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}]> : tensor<210xf32>
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<32> : tensor<i32>
  %c_2 = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %[[RESHARD]], %iterArg_2 = %c)
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c) : tensor<210xf32>, tensor<i32> attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>, <@mesh, []>]>}
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.dot %iterArg, %iterArg : (tensor<210xf32>, tensor<210xf32>) -> tensor<f32>
    %4 = stablehlo.compare  LT, %3, %c_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %2, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %iterArg <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
    %3 = stablehlo.negate %iterArg {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    %4 = stablehlo.add %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}: tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]], %{{.*}} : tensor<210xf32>, tensor<i32>
    stablehlo.return %4, %2 : tensor<210xf32>, tensor<i32>
  }
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %[[WHILE]]#0 <@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.add %arg0, %[[RESHARD]]
  %1 = stablehlo.add %arg0, %0#0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @while_missing_sharding
func.func @while_missing_sharding(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32>) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<32> : tensor<i32>
  %c_2 = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}]> : tensor<210xf32>
  // CHECK-NEXT stablehlo.while(%iterArg = %[[RESHARD]], %iterArg_2 = %c)
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c) : tensor<210xf32>, tensor<i32>
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.dot %iterArg, %iterArg : (tensor<210xf32>, tensor<210xf32>) -> tensor<f32>
    %4 = stablehlo.compare  LT, %3, %c_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %2, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %iterArg <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
    %3 = stablehlo.negate %iterArg {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    %4 = stablehlo.add %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}: tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]], %{{.*}} : tensor<210xf32>, tensor<i32>
    stablehlo.return %4, %2 : tensor<210xf32>, tensor<i32>
  }
  %1 = stablehlo.negate %0#0 : tensor<210xf32>
  return %1: tensor<210xf32>
}

// CHECK-LABEL: func @while_fully_replicated_everywhere
func.func @while_fully_replicated_everywhere(%arg0: tensor<210xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<210xf32>) {
  // CHECK-NOT: sdy.reshard
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<32> : tensor<i32>
  %c_2 = stablehlo.constant dense<0.0> : tensor<f32>
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c) : tensor<210xf32>, tensor<i32>
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.dot %iterArg, %iterArg : (tensor<210xf32>, tensor<210xf32>) -> tensor<f32>
    %4 = stablehlo.compare  LT, %3, %c_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %2, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
    %3 = stablehlo.negate %iterArg : tensor<210xf32>
    %4 = stablehlo.add %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<210xf32>
    stablehlo.return %4, %2 : tensor<210xf32>, tensor<i32>
  }
  %1 = stablehlo.negate %0#0 : tensor<210xf32>
  return %1: tensor<210xf32>
}

// CHECK-LABEL: func @optimization_barrier
func.func @optimization_barrier(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} %[[RESHARD]]
  %1 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} %arg0 : tensor<210xf32>
  %2 = stablehlo.negate %1 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %2 : tensor<210xf32>
}
