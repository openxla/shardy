// RUN: sdy_opt %s -sdy-constant-or-scalar-splitter 2>&1 | FileCheck %s

// CHECK-LABEL: func @constant_with_unregistered_attr
func.func @constant_with_unregistered_attr() -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant {foo} dense<1.000000e+00>
  // CHECK-NEXT: return %[[CONST]]
  %0 = stablehlo.constant {foo} dense<1.000000e+00> : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_arg_is_not_constant
func.func @func_arg_is_not_constant(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[CONST]]
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[ADD]], %arg1
  // CHECK-NEXT: return %[[ADD]], %[[DOT_GENERAL]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<8x16xf32>
  %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  return %1, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_multiple_users_func_no_argument
func.func @constant_multiple_users_func_no_argument() -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[CONST_1]]
  // CHECK-NEXT: return %[[CONST_0]], %[[NEGATE]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.negate %0 : tensor<8x16xf32>
  return %0, %1: tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_multiple_users
func.func @constant_multiple_users(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_1]], %[[DOT_GENERAL]]
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_multiple_users_within_named_computation
func.func @constant_multiple_users_within_named_computation(%arg0: tensor<16x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<16x16xf32>) {
  // CHECK-NEXT:   %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT:   %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT:   %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT:   %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %arg1
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[CONST_1]], %[[DOT_GENERAL]]
  // CHECK-NEXT:   sdy.return %[[CONST_2]], %[[ADD]]
  // CHECK-NEXT: } : (tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  // CHECK-NEXT: %[[MULTIPLY:.*]] = stablehlo.multiply %[[NC]]#0, %[[NC]]#1
  // CHECK-NEXT: return %[[MULTIPLY]]
  %0:2 = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<16x16xf32>) {
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
    %3 = stablehlo.add %1, %2 : tensor<8x16xf32>
    sdy.return %1, %3 : tensor<8x16xf32>, tensor<8x16xf32>
  } : (tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  %4 = stablehlo.multiply %0#0, %0#1 : tensor<8x16xf32>
  return %4 : tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_multiple_users_within_named_computation_with_only_constant_ops
func.func @constant_multiple_users_within_named_computation_with_only_constant_ops(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%[[ABS]]) (%arg1: tensor<8x16xf32>) {
  // CHECK-NEXT:   %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT:   %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT:   %[[NEGATE_0:.*]] = stablehlo.negate %[[CONST_0]]
  // CHECK-NEXT:   %[[NEGATE_1:.*]] = stablehlo.negate %[[CONST_1]]
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[NEGATE_0]], %arg1
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %[[NEGATE_1]], %[[ADD]]
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]]
  // CHECK-NEXT: } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[NEGATE_2:.*]] = stablehlo.negate %[[NC]]
  // CHECK-NEXT: return %[[NEGATE_2]]
  %0 = stablehlo.abs %arg0 : tensor<8x16xf32>
  %1 = sdy.named_computation<"foo">(%0) (%arg1: tensor<8x16xf32>) {
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
    %3 = stablehlo.negate %2 : tensor<8x16xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<8x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x16xf32>
    sdy.return %5 : tensor<8x16xf32>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %6 = stablehlo.negate %1 : tensor<8x16xf32>
  return %6 : tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_to_named_computation_with_only_constant_ops
func.func @constant_to_named_computation_with_only_constant_ops(%arg0: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%[[CONST]], %arg0) (%arg1: tensor<8x16xf32>, %arg2: tensor<8x16xf32>) {
  // CHECK-NEXT:   %[[NEGATE:.*]] = stablehlo.negate %arg1
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[NEGATE]], %arg2
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %[[NEGATE]], %[[ADD]]
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]]
  // CHECK-NEXT: } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ABS_0:.*]] = stablehlo.abs %[[NC]]
  // CHECK-NEXT: %[[ABS_1:.*]] = stablehlo.abs %[[NC]]
  // CHECK-NEXT: return %[[ABS_0]], %[[ABS_1]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = sdy.named_computation<"foo">(%0, %arg0) (%arg1: tensor<8x16xf32>, %arg2: tensor<8x16xf32>) {
    %2 = stablehlo.negate %arg1 : tensor<8x16xf32>
    %3 = stablehlo.add %2, %arg2 : tensor<8x16xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<8x16xf32>
    sdy.return %4 : tensor<8x16xf32>
  } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %5 = stablehlo.abs %1 : tensor<8x16xf32>
  %6 = stablehlo.abs %1 : tensor<8x16xf32>
  return %5, %6 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_multiple_users_within_named_computation_with_no_arguments_and_with_only_constant_ops
func.func @constant_multiple_users_within_named_computation_with_no_arguments_and_with_only_constant_ops() -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">()
  // CHECK-NEXT:   %[[CONST:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT:   %[[NEGATE:.*]] = stablehlo.negate %[[CONST]]
  // CHECK-NEXT:   sdy.return %[[NEGATE]]
  // CHECK-NEXT: } : () -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ABS_0:.*]] = stablehlo.abs %[[NC]]
  // CHECK-NEXT: %[[ABS_1:.*]] = stablehlo.abs %[[NC]]
  // CHECK-NEXT: return %[[ABS_0]], %[[ABS_1]]
  // TODO(enver): The named computation should be splitted.
  %0 = sdy.named_computation<"foo">() () {
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
    %2 = stablehlo.negate %1 : tensor<8x16xf32>
    sdy.return %2 : tensor<8x16xf32>
  } : () -> tensor<8x16xf32>
  %3 = stablehlo.abs %0 : tensor<8x16xf32>
  %4 = stablehlo.abs %0 : tensor<8x16xf32>
  return %3, %4 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_to_named_computation_with_one_argument_and_with_only_constant_ops
func.func @constant_to_named_computation_with_one_argument_and_with_only_constant_ops() -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%[[CONST]]) (%arg0: tensor<8x16xf32>) {
  // CHECK-NEXT:   %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg0, %[[NEGATE]]
  // CHECK-NEXT:   sdy.return %[[ADD]]
  // CHECK-NEXT: } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ABS_0:.*]] = stablehlo.abs %[[NC]]
  // CHECK-NEXT: %[[ABS_1:.*]] = stablehlo.abs %[[NC]]
  // CHECK-NEXT: return %[[ABS_0]], %[[ABS_1]]
  // TODO(enver): The named computation should be splitted.
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = sdy.named_computation<"foo">(%0) (%arg0: tensor<8x16xf32>) {
    %2 = stablehlo.negate %arg0 : tensor<8x16xf32>
    %3 = stablehlo.add %arg0, %2: tensor<8x16xf32>
    sdy.return %3 : tensor<8x16xf32>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %4 = stablehlo.abs %1 : tensor<8x16xf32>
  %5 = stablehlo.abs %1 : tensor<8x16xf32>
  return %4, %5 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_multiple_users_one_to_named_computation_with_one_argument_and_with_only_constant_ops
func.func @constant_multiple_users_one_to_named_computation_with_one_argument_and_with_only_constant_ops() -> (tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%[[CONST_0]]) (%arg0: tensor<8x16xf32>) {
  // CHECK-NEXT:   %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg0, %[[NEGATE]]
  // CHECK-NEXT:   sdy.return %[[ADD]]
  // CHECK-NEXT: } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ABS_0:.*]] = stablehlo.abs %[[NC]]
  // CHECK-NEXT: %[[ABS_1:.*]] = stablehlo.abs %[[NC]]
  // CHECK-NEXT: return %[[CONST_1]], %[[ABS_0]], %[[ABS_1]]
  // TODO(enver): The named computation should be splitted.
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = sdy.named_computation<"foo">(%0) (%arg0: tensor<8x16xf32>) {
    %2 = stablehlo.negate %arg0 : tensor<8x16xf32>
    %3 = stablehlo.add %arg0, %2: tensor<8x16xf32>
    sdy.return %3 : tensor<8x16xf32>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %4 = stablehlo.abs %1 : tensor<8x16xf32>
  %5 = stablehlo.abs %1 : tensor<8x16xf32>
  return %0, %4, %5 : tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_multiple_users_one_to_nested_named_computations_with_one_argument_and_with_only_constant_ops
func.func @constant_multiple_users_one_to_nested_named_computations_with_one_argument_and_with_only_constant_ops() -> (tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[NC_0:.*]] = sdy.named_computation<"foo">(%[[CONST_0]]) (%arg0: tensor<8x16xf32>) {
  // CHECK-NEXT:   %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT:   %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT:   %[[NC_1:.*]] = sdy.named_computation<"bar">(%[[NEGATE]], %[[CONST_2]]) (%arg1: tensor<8x16xf32>, %arg2: tensor<8x16xf32>) {
  // CHECK-NEXT:     %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2
  // CHECK-NEXT:     sdy.return %[[MULTIPLY]]
  // CHECK-NEXT:   } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg0, %[[NC_1]]
  // CHECK-NEXT:   sdy.return %[[ADD]]
  // CHECK-NEXT: } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ABS_0:.*]] = stablehlo.abs %[[NC_0]]
  // CHECK-NEXT: %[[ABS_1:.*]] = stablehlo.abs %[[NC_0]]
  // CHECK-NEXT: return %[[CONST_1]], %[[ABS_0]], %[[ABS_1]]
  // TODO(enver): The named computation should be splitted.
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = sdy.named_computation<"foo">(%0) (%arg0: tensor<8x16xf32>) {
    %2 = stablehlo.negate %arg0 : tensor<8x16xf32>
    %3 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
    %4 = sdy.named_computation<"bar">(%2, %3) (%arg1: tensor<8x16xf32>, %arg2: tensor<8x16xf32>) {
      %5 = stablehlo.multiply %arg1, %arg2 : tensor<8x16xf32>
      sdy.return %5: tensor<8x16xf32>
    } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
    %6 = stablehlo.add %arg0, %4: tensor<8x16xf32>
    sdy.return %6 : tensor<8x16xf32>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %7 = stablehlo.abs %1 : tensor<8x16xf32>
  %8 = stablehlo.abs %1 : tensor<8x16xf32>
  return %0, %7, %8 : tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_multiple_users_one_to_nested_named_computations_with_one_argument_and_inner_named_computation_is_not_constant_like
func.func @constant_multiple_users_one_to_nested_named_computations_with_one_argument_and_inner_named_computation_is_not_constant_like() -> (tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[NC_0:.*]] = sdy.named_computation<"foo">(%[[CONST_0]]) (%arg0: tensor<8x16xf32>) {
  // CHECK-NEXT:   %[[NEGATE:.*]] = stablehlo.negate %arg0
  // CHECK-NEXT:   %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT:   %[[NC_1:.*]] = sdy.named_computation<"bar">(%[[NEGATE]], %[[CONST_2]]) (%arg1: tensor<8x16xf32>, %arg2: tensor<16x16xf32>) {
  // CHECK-NEXT:     %[[DOT:.*]] = stablehlo.dot %arg1, %arg2
  // CHECK-NEXT:     sdy.return %[[DOT]]
  // CHECK-NEXT:   } : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg0, %[[NC_1]]
  // CHECK-NEXT:   sdy.return %[[ADD]]
  // CHECK-NEXT: } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ABS_0:.*]] = stablehlo.abs %[[NC_0]]
  // CHECK-NEXT: %[[ABS_1:.*]] = stablehlo.abs %[[NC_0]]
  // CHECK-NEXT: return %[[CONST_1]], %[[ABS_0]], %[[ABS_1]]
  // TODO(enver): The named computation should be splitted.
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = sdy.named_computation<"foo">(%0) (%arg0: tensor<8x16xf32>) {
    %2 = stablehlo.negate %arg0 : tensor<8x16xf32>
    %3 = stablehlo.constant dense<1.000000e+00> : tensor<16x16xf32>
    %4 = sdy.named_computation<"bar">(%2, %3) (%arg1: tensor<8x16xf32>, %arg2: tensor<16x16xf32>) {
      %5 = stablehlo.dot %arg1, %arg2 : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
      sdy.return %5: tensor<8x16xf32>
    } : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
    %6 = stablehlo.add %arg0, %4: tensor<8x16xf32>
    sdy.return %6 : tensor<8x16xf32>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %7 = stablehlo.abs %1 : tensor<8x16xf32>
  %8 = stablehlo.abs %1 : tensor<8x16xf32>
  return %0, %7, %8 : tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @scalar_constant_multiple_users_simple
func.func @scalar_constant_multiple_users_simple(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST]], %arg0
  // CHECK-NEXT: return %[[CONST]], %[[ADD]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.add %0, %arg0 : tensor<f32>
  return %0, %1 : tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func @non_scalar_constant_multiple_users_simple
func.func @non_scalar_constant_multiple_users_simple(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  // CHECK-NEXT: %[[CONST1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST1]], %arg0
  // CHECK-NEXT: return %[[CONST2]], %[[ADD]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %1 = stablehlo.add %0, %arg0 : tensor<2xf32>
  return %0, %1 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func @constant_sub_computation_multiple_users
func.func @constant_sub_computation_multiple_users(%arg0: tensor<5x8xi32>) -> (tensor<4x5xi32>, tensor<4x8xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<2>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<2>
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[IOTA_0]], %[[IOTA_0]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[IOTA_1]], %[[IOTA_1]]
  // CHECK-NEXT: %[[MAX_0:.*]] = stablehlo.maximum %[[ADD_0]], %[[CONST_0]]
  // CHECK-NEXT: %[[MAX_1:.*]] = stablehlo.maximum %[[ADD_1]], %[[CONST_1]]
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[MAX_0]], %arg0
  // CHECK-NEXT: return %[[MAX_1]], %[[DOT_GENERAL]]
  %0 = stablehlo.iota dim = 0 : tensor<4x5xi32>
  %1 = stablehlo.constant dense<2> : tensor<4x5xi32>
  %2 = stablehlo.add %0, %0 : tensor<4x5xi32>
  %3 = stablehlo.maximum %2, %1 : tensor<4x5xi32>
  %4 = stablehlo.dot_general %3, %arg0, contracting_dims = [1] x [0] : (tensor<4x5xi32>, tensor<5x8xi32>) -> tensor<4x8xi32>
  return %3, %4 : tensor<4x5xi32>, tensor<4x8xi32>
}

// CHECK-LABEL: func @constant_multiple_uses_by_same_op
func.func @constant_multiple_uses_by_same_op() -> (tensor<8x16xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %[[CONST_1]]
  // CHECK-NEXT: return %[[CONST_2]], %[[DOT_GENERAL]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [1] : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x8xf32>
  return %0, %1 : tensor<8x16xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @non_constant_broadcast_multiple_users
func.func @non_constant_broadcast_multiple_users(%arg0: tensor<4x5xf32>, %arg1: tensor<5xf32>) -> (tensor<5x8xf32>, tensor<4x8xf32>) {
  // CHECK-NEXT: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %[[BROADCAST]]
  // CHECK-NEXT: return %[[BROADCAST]], %[[DOT_GENERAL]]
  %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<5xf32>) -> tensor<5x8xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<4x5xf32>, tensor<5x8xf32>) -> tensor<4x8xf32>
  return %0, %1 : tensor<5x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @constant_broadcast_multiple_users
func.func @constant_broadcast_multiple_users(%arg0: tensor<5x8xi32>) -> (tensor<4x5xi32>, tensor<4x8xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[IOTA_0]], %[[IOTA_0]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[IOTA_1]], %[[IOTA_1]]
  // CHECK-NEXT: %[[BROADCAST_0:.*]] = stablehlo.broadcast_in_dim %[[ADD_0]]
  // CHECK-NEXT: %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %[[ADD_1]]
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[BROADCAST_0]], %arg0
  // CHECK-NEXT: return %[[BROADCAST_1]], %[[DOT_GENERAL]]
  %0 = stablehlo.iota dim = 0 : tensor<5xi32>
  %1 = stablehlo.add %0, %0 : tensor<5xi32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<5xi32>) -> tensor<4x5xi32>
  %3 = stablehlo.dot_general %2, %arg0, contracting_dims = [1] x [0] : (tensor<4x5xi32>, tensor<5x8xi32>) -> tensor<4x8xi32>
  return %2, %3 : tensor<4x5xi32>, tensor<4x8xi32>
}

// CHECK-LABEL: func @multiple_broadcasts_using_the_same_const_sub_computation
func.func @multiple_broadcasts_using_the_same_const_sub_computation(%arg0: tensor<5x8xi32>) -> (tensor<3x5xi32>, tensor<2x4xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_2:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[IOTA_0]], %[[IOTA_0]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[IOTA_1]], %[[IOTA_1]]
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[IOTA_2]], %[[IOTA_2]]
  // CHECK-NEXT: %[[BROADCAST_0:.*]] = stablehlo.broadcast_in_dim %[[ADD_0]], dims = [1] : (tensor<5xi32>) -> tensor<2x5xi32>
  // CHECK-NEXT: %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %[[ADD_2]], dims = [1] : (tensor<5xi32>) -> tensor<3x5xi32>
  // CHECK-NEXT: %[[BROADCAST_2:.*]] = stablehlo.broadcast_in_dim %[[ADD_1]], dims = [0] : (tensor<5xi32>) -> tensor<5x4xi32>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[BROADCAST_0]], %[[BROADCAST_2]]
  // CHECK-NEXT: return %[[BROADCAST_1]], %[[DOT_GENERAL]]
  %0 = stablehlo.iota dim = 0 : tensor<5xi32>
  %1 = stablehlo.add %0, %0 : tensor<5xi32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<5xi32>) -> tensor<2x5xi32>
  %3 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<5xi32>) -> tensor<3x5xi32>
  %4 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<5xi32>) -> tensor<5x4xi32>
  %5 = stablehlo.dot_general %2, %4, contracting_dims = [1] x [0] : (tensor<2x5xi32>, tensor<5x4xi32>) -> tensor<2x4xi32>
  return %3, %5 : tensor<3x5xi32>, tensor<2x4xi32>
}

// CHECK-LABEL: func @constant_reshape_multiple_users
func.func @constant_reshape_multiple_users() -> (tensor<5x2xi32>, tensor<5x2xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %[[IOTA_0]]
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[IOTA_1]]
  // CHECK-NEXT: return %[[RESHAPE_0]], %[[RESHAPE_1]]
  %0 = stablehlo.iota dim = 0 : tensor<10xi32>
  %1 = stablehlo.reshape %0 : (tensor<10xi32>) -> tensor<5x2xi32>
  return %1, %1 : tensor<5x2xi32>, tensor<5x2xi32>
}

// CHECK-LABEL: func @constant_slice_multiple_users
func.func @constant_slice_multiple_users(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[SLICE_0:.*]] = stablehlo.slice %[[IOTA_0]]
  // CHECK-NEXT: %[[SLICE_1:.*]] = stablehlo.slice %[[IOTA_1]]
  // CHECK-NEXT: %[[SLICE_2:.*]] = stablehlo.slice %[[IOTA_0]]
  // CHECK-NEXT: %[[SLICE_3:.*]] = stablehlo.slice %[[IOTA_1]]
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[SLICE_0]], %[[SLICE_2]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[SLICE_1]], %[[SLICE_3]]
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[ADD_0]], %arg0
  // CHECK-NEXT: return %[[ADD_1]], %[[ADD_2]]
  %0 = stablehlo.iota dim = 0 : tensor<10xi32>
  %1 = stablehlo.slice %0 [0:8]: (tensor<10xi32>) -> tensor<8xi32>
  %2 = stablehlo.slice %0 [2:10]: (tensor<10xi32>) -> tensor<8xi32>
  %3 = stablehlo.add %1, %2 : tensor<8xi32>
  %4 = stablehlo.add %3, %arg0 : tensor<8xi32>
  return %3, %4 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func @splits_parts_of_const_sub_computation
func.func @splits_parts_of_const_sub_computation(%arg0: tensor<5x8xi32>) -> (tensor<4x5xi32>, tensor<4x5xi32>, tensor<4x8xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_2:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant dense<2>
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[IOTA_0]], %[[IOTA_0]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[IOTA_2]], %[[IOTA_2]]
  // CHECK-NEXT: %[[MAX:.*]] = stablehlo.maximum %[[ADD_0]], %[[CONST]]
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[MAX]], %arg0
  // CHECK-NEXT: return %[[IOTA_1]], %[[ADD_1]], %[[DOT_GENERAL]]
  %0 = stablehlo.iota dim = 0 : tensor<4x5xi32>
  %1 = stablehlo.constant dense<2> : tensor<4x5xi32>
  %2 = stablehlo.add %0, %0 : tensor<4x5xi32>
  %3 = stablehlo.maximum %2, %1 : tensor<4x5xi32>
  %4 = stablehlo.dot_general %3, %arg0, contracting_dims = [1] x [0] : (tensor<4x5xi32>, tensor<5x8xi32>) -> tensor<4x8xi32>
  return %0, %2, %4 : tensor<4x5xi32>, tensor<4x5xi32>, tensor<4x8xi32>
}

// CHECK-LABEL: func @splits_sdy_constants
func.func @splits_sdy_constants(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_1]], %[[DOT_GENERAL]]
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
  %0 = sdy.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @splits_sharding_groups
func.func @splits_sharding_groups(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // Each intermediate result is included in a distinct sharding group. Post
  // splitting, we verify that all subcomputation results corresponding to an
  // original unsplit result are included in the same sharding group.
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: sdy.sharding_group %[[CONST_0]] group_id=0
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: sdy.sharding_group %[[CONST_1]] group_id=0
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: sdy.sharding_group %[[CONST_2]] group_id=0
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %arg0
  // CHECK-NEXT: sdy.sharding_group %[[DOT_GENERAL]] group_id=1
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_1]], %[[DOT_GENERAL]]
  // CHECK-NEXT: sdy.sharding_group %[[ADD]] group_id=2
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  sdy.sharding_group %0 group_id=0 : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  sdy.sharding_group %1 group_id=1 : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  sdy.sharding_group %2 group_id=2 : tensor<8x16xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @splits_const_subexpr_with_sharding_group
func.func @splits_const_subexpr_with_sharding_group(%arg0: tensor<4x8xi32>) -> (tensor<4x8xi32>, tensor<4x8xi32>) {
  // CHECK-NEXT: sdy.sharding_group %arg0 group_id=0
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: sdy.sharding_group %[[IOTA_0]] group_id=0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: sdy.sharding_group %[[IOTA_1]] group_id=0
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<2>
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[IOTA_0]], %[[IOTA_0]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[IOTA_1]], %[[IOTA_1]]
  // CHECK-NEXT: %[[MAX_0:.*]] = stablehlo.maximum %[[ADD_1]], %[[CONST_0]]
  sdy.sharding_group %arg0 group_id=0 : tensor<4x8xi32>
  %0 = stablehlo.iota dim = 0 : tensor<4x8xi32>
  sdy.sharding_group %0 group_id=0 : tensor<4x8xi32>
  %1 = stablehlo.constant dense<2> : tensor<4x8xi32>
  %2 = stablehlo.add %0, %0 : tensor<4x8xi32>
  %3 = stablehlo.maximum %2, %1 : tensor<4x8xi32>
  return %2, %3 : tensor<4x8xi32>, tensor<4x8xi32>
}

// CHECK-LABEL: func @does_not_split_broadcast_on_non_scalar_input
func.func @does_not_split_broadcast_on_non_scalar_input(%arg0: tensor<2xf32>) -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.negate %0 : tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.abs %0 : tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.multiply %1, %2 : tensor<2x64xf32>
  // CHECK-NEXT: return %3 :  tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  %1 = stablehlo.negate %0 : tensor<2x64xf32>
  %2 = stablehlo.abs %0 : tensor<2x64xf32>
  %3 = stablehlo.multiply %1, %2 : tensor<2x64xf32>
  return %3 :  tensor<2x64xf32>
}

// CHECK-LABEL: func @does_not_split_broadcast_on_one_dimensional_input_of_size_one
func.func @does_not_split_broadcast_on_one_dimensional_input_of_size_one(%arg0: tensor<1xf32>) -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.negate %0 : tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.abs %0 : tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.multiply %1, %2 : tensor<2x64xf32>
  // CHECK-NEXT: return %3 :  tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf32>) -> tensor<2x64xf32>
  %1 = stablehlo.negate %0 : tensor<2x64xf32>
  %2 = stablehlo.abs %0 : tensor<2x64xf32>
  %3 = stablehlo.multiply %1, %2 : tensor<2x64xf32>
  return %3 :  tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_broadcast_on_scalar_simple
func.func @splits_broadcast_on_scalar_simple(%arg0: tensor<f32>) -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.negate %0 : tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.abs %1 : tensor<2x64xf32>
  // CHECK-NEXT: %4 = stablehlo.multiply %2, %3 : tensor<2x64xf32>
  // CHECK-NEXT: return %4 :  tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.negate %0 : tensor<2x64xf32>
  %2 = stablehlo.abs %0 : tensor<2x64xf32>
  %3 = stablehlo.multiply %1, %2 : tensor<2x64xf32>
  return %3 :  tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_multiple_broadcast_on_scalar_use_same_scalar
func.func @splits_multiple_broadcast_on_scalar_use_same_scalar(%arg0: tensor<f32>) -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.add %0, %1 : tensor<2x64xf32>
  // CHECK-NEXT: return %2 :  tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %2 = stablehlo.add %0, %1 : tensor<2x64xf32>
  return %2 :  tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_multiple_broadcasts_on_scalar_non_constant
func.func @splits_multiple_broadcasts_on_scalar_non_constant(%arg0: tensor<f32>) -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.negate %0 : tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.abs %1 : tensor<2x64xf32>
  // CHECK-NEXT: %4 = stablehlo.multiply %2, %3 : tensor<2x64xf32>
  // CHECK-NEXT: %5 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %6 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %7 = stablehlo.negate %5 : tensor<2x64xf32>
  // CHECK-NEXT: %8 = stablehlo.abs %6 : tensor<2x64xf32>
  // CHECK-NEXT: %9 = stablehlo.multiply %7, %8 : tensor<2x64xf32>
  // CHECK-NEXT: %10 = stablehlo.add %4, %9 : tensor<2x64xf32>
  // CHECK-NEXT: return %10 :  tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.negate %0 : tensor<2x64xf32>
  %2 = stablehlo.abs %0 : tensor<2x64xf32>
  %3 = stablehlo.multiply %1, %2 : tensor<2x64xf32>
  %4 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %5 = stablehlo.negate %4 : tensor<2x64xf32>
  %6 = stablehlo.abs %4 : tensor<2x64xf32>
  %7 = stablehlo.multiply %5, %6 : tensor<2x64xf32>
  %8 = stablehlo.add %3, %7 : tensor<2x64xf32>
  return %8 :  tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_broadcast_on_scalar_non_constant_use_and_itself_on_same_op
func.func @splits_broadcast_on_scalar_non_constant_use_and_itself_on_same_op(%arg0: tensor<f32>) -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.negate %0 : tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.multiply %1, %2 : tensor<2x64xf32>
  // CHECK-NEXT: return %3 :  tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.negate %0 : tensor<2x64xf32>
  %2 = stablehlo.multiply %0, %1 : tensor<2x64xf32>
  return %2 :  tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_broadcast_on_scalar_non_constant_one_of_multiple_use_is_func_return
func.func @splits_broadcast_on_scalar_non_constant_one_of_multiple_use_is_func_return(%arg0: tensor<f32>) -> (tensor<2x64xf32>, tensor<2x64xf32>) {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.negate %0 : tensor<2x64xf32>
  // CHECK-NEXT: return %1, %2 : tensor<2x64xf32>, tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.negate %0 : tensor<2x64xf32>
  return %0, %1 : tensor<2x64xf32>, tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_broadcast_on_scalar_non_constant_all_uses_on_same_op
func.func @splits_broadcast_on_scalar_non_constant_all_uses_on_same_op(%arg0: tensor<f32>) -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.add %0, %1 : tensor<2x64xf32>
  // CHECK-NEXT: return %2 :  tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.add %0, %0 : tensor<2x64xf32>
  return %1 :  tensor<2x64xf32>
}

// CHECK-LABEL: func @does_not_split_broadcast_single_use
func.func @does_not_split_broadcast_single_use(%arg0: tensor<f32>) -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.negate %0 : tensor<2x64xf32>
  // CHECK-NEXT: return %1 :  tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.negate %0 : tensor<2x64xf32>
  return %1 :  tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_broadcast_on_scalar_non_constant_multiple_uses
func.func @splits_broadcast_on_scalar_non_constant_multiple_uses(%arg0: tensor<f32>) -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %4 = stablehlo.add %0, %1 : tensor<2x64xf32>
  // CHECK-NEXT: %5 = stablehlo.negate %2 : tensor<2x64xf32>
  // CHECK-NEXT: %6 = stablehlo.divide %3, %5 : tensor<2x64xf32>
  // CHECK-NEXT: %7 = stablehlo.multiply %4, %6 : tensor<2x64xf32>
  // CHECK-NEXT: return %7 :  tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.add %0, %0 : tensor<2x64xf32>
  %2 = stablehlo.negate %0 : tensor<2x64xf32>
  %3 = stablehlo.divide %0, %2 : tensor<2x64xf32>
  %4 = stablehlo.multiply %1, %3 : tensor<2x64xf32>
  return %4 :  tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_both_constant_and_broadcast_on_scalar_op_result
func.func @splits_both_constant_and_broadcast_on_scalar_op_result() -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = sdy.constant dense<0.000000e+00> : tensor<32xf32>
  // CHECK-NEXT: %1 = sdy.constant dense<0.000000e+00> : tensor<32xf32>
  // CHECK-NEXT: %2 = stablehlo.dot %0, %1 : (tensor<32xf32>, tensor<32xf32>) -> tensor<f32>
  // CHECK-NEXT: %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %5 = stablehlo.add %3, %4 : tensor<2x64xf32>
  // CHECK-NEXT: return %5 : tensor<2x64xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<32xf32>
  %1 = stablehlo.dot %0, %0 : (tensor<32xf32>, tensor<32xf32>) -> tensor<f32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %3 = stablehlo.add %2, %2 : tensor<2x64xf32>
  return %3 : tensor<2x64xf32>
}

// CHECK-LABEL: func @does_not_split_broadcast_on_scalar_constant_diamond
func.func @does_not_split_broadcast_on_scalar_constant_diamond() -> tensor<2x64xf32> {
  // CHECK-NEXT: %0 = sdy.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.negate %1 : tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.abs %1 : tensor<2x64xf32>
  // CHECK-NEXT: %4 = stablehlo.multiply %2, %3 : tensor<2x64xf32>
  // CHECK-NEXT: return %4 : tensor<2x64xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %2 = stablehlo.negate %1 : tensor<2x64xf32>
  %3 = stablehlo.abs %1 : tensor<2x64xf32>
  %4 = stablehlo.multiply %2, %3 : tensor<2x64xf32>
  return %4 : tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_broadcast_on_scalar_constant_non_diamond
func.func @splits_broadcast_on_scalar_constant_non_diamond() -> (tensor<2x64xf32>, tensor<2x64xf32>) {
  // CHECK-NEXT: %0 = sdy.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.negate %2 : tensor<2x64xf32>
  // CHECK-NEXT: %4 = stablehlo.abs %1 : tensor<2x64xf32>
  // CHECK-NEXT: %5 = stablehlo.abs %2 : tensor<2x64xf32>
  // CHECK-NEXT: %6 = stablehlo.multiply %3, %5 : tensor<2x64xf32>
  // CHECK-NEXT: return %4, %6 : tensor<2x64xf32>, tensor<2x64xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %2 = stablehlo.negate %1 : tensor<2x64xf32>
  %3 = stablehlo.abs %1 : tensor<2x64xf32>
  %4 = stablehlo.multiply %2, %3 : tensor<2x64xf32>
  return %3, %4 : tensor<2x64xf32>, tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_broadcast_on_scalar_constant_diamond_also_used_outside_diamond
func.func @splits_broadcast_on_scalar_constant_diamond_also_used_outside_diamond() -> (tensor<2x64xf32>, tensor<2x64xf32>) {
  // CHECK-NEXT: %0 = sdy.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.negate %2 : tensor<2x64xf32>
  // CHECK-NEXT: %4 = stablehlo.abs %2 : tensor<2x64xf32>
  // CHECK-NEXT: %5 = stablehlo.multiply %3, %4 : tensor<2x64xf32>
  // CHECK-NEXT: return %1, %5 : tensor<2x64xf32>, tensor<2x64xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %2 = stablehlo.negate %1 : tensor<2x64xf32>
  %3 = stablehlo.abs %1 : tensor<2x64xf32>
  %4 = stablehlo.multiply %2, %3 : tensor<2x64xf32>
  return %1, %4 : tensor<2x64xf32>, tensor<2x64xf32>
}

// CHECK-LABEL: func @splits_broadcast_on_non_scalar_constant
func.func @splits_broadcast_on_non_scalar_constant() -> (tensor<2x64xf32>, tensor<2x64xf32>) {
  // CHECK-NEXT: %0 = sdy.constant dense<0.000000e+00> : tensor<2xf32>
  // CHECK-NEXT: %1 = sdy.constant dense<0.000000e+00> : tensor<2xf32>
  // CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %4 = stablehlo.negate %3 : tensor<2x64xf32>
  // CHECK-NEXT: %5 = stablehlo.abs %3 : tensor<2x64xf32>
  // CHECK-NEXT: %6 = stablehlo.multiply %4, %5 : tensor<2x64xf32>
  // CHECK-NEXT: return %2, %6 : tensor<2x64xf32>, tensor<2x64xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<2xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  %2 = stablehlo.negate %1 : tensor<2x64xf32>
  %3 = stablehlo.abs %1 : tensor<2x64xf32>
  %4 = stablehlo.multiply %2, %3 : tensor<2x64xf32>
  return %1, %4 : tensor<2x64xf32>, tensor<2x64xf32>
}

// CHECK-LABEL: func @broadcast_on_scalar_constant_propagates_constant_expansion
func.func @broadcast_on_scalar_constant_propagates_constant_expansion() -> (tensor<2x64xf32>, tensor<2x64xf32>) {
  // CHECK-NEXT: %0 = sdy.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.negate %1 : tensor<2x64xf32>
  // CHECK-NEXT: %4 = stablehlo.negate %2 : tensor<2x64xf32>
  // CHECK-NEXT: %5 = stablehlo.add %1, %3 : tensor<2x64xf32>
  // CHECK-NEXT: %6 = stablehlo.add %2, %4 : tensor<2x64xf32>
  // CHECK-NEXT: return %5, %6 : tensor<2x64xf32>, tensor<2x64xf32>
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %2 = stablehlo.negate %1 : tensor<2x64xf32>
  %3 = stablehlo.add %1, %2 : tensor<2x64xf32>
  return %3, %3 : tensor<2x64xf32>, tensor<2x64xf32>
}

// CHECK-LABEL: func @broadcast_on_scalar_non_constant_does_not_propagate_constant_expansion
func.func @broadcast_on_scalar_non_constant_does_not_propagate_constant_expansion(%arg0: tensor<f32>) -> (tensor<2x64xf32>, tensor<2x64xf32>) {
  // CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.negate %0 : tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.add %1, %2 : tensor<2x64xf32>
  // CHECK-NEXT: return %3, %3 : tensor<2x64xf32>, tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.negate %0 : tensor<2x64xf32>
  %2 = stablehlo.add %0, %1 : tensor<2x64xf32>
  return %2, %2 : tensor<2x64xf32>, tensor<2x64xf32>
}
