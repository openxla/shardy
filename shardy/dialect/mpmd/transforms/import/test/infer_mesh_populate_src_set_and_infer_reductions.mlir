// RUN: mpmd_opt %s -mpmd-infer-mesh-populate-src-set 2>&1 | FileCheck --implicit-check-not src_set --implicit-check-not mpmd.reduce %s

!m1_4x8 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!m2_4x8 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
!m3_4x8 = !mpmd.mesh_tensor<"m3", tensor<4x8xf32>>

!m1_4x1x8 = !mpmd.mesh_tensor<"m1", tensor<4x1x8xf32>>
!m2_4x1x8 = !mpmd.mesh_tensor<"m2", tensor<4x1x8xf32>>

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>>

// The main tests are checked in `mpmd_infer_mesh_populate_src_set.mlir`. This
// only verifies the infer_reduce logic.
//
// The `--implicit-check-not src_set` on FileCheck means that the text "src_set"
// and "mpmd.reduce" is only allowed when explicitly specified in a CHECK.


// CHECK-LABEL: func @simple_reduce
func.func @simple_reduce(%arg0: !m1_4x8, %arg1: !m2_4x8)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %1 = mpmd.unassign %arg0 : (!m1_4x8) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  %2 = mpmd.unassign %arg1 : (!m2_4x8) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  // The src_sets would normally be empty, but we parse the reduce.
  %3 = stablehlo.maximum %1, %2 : tensor<4x8xf32> // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = {{.*}}"m1", "m2">}
  %4 = stablehlo.maximum %0, %3 : tensor<4x8xf32> // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = {{.*}}"m1", "m2">}
  func.return %4 : tensor<4x8xf32>
}

// CHECK-LABEL: func @simple_reduce_chain
func.func @simple_reduce_chain(%arg0: !m1_4x8, %arg1: !m2_4x8, %arg2: !m3_4x8)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  %1 = mpmd.unassign %arg0 : (!m1_4x8) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  %2 = mpmd.unassign %arg1 : (!m2_4x8) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  %3 = mpmd.unassign %arg2 : (!m3_4x8) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m3">
  %4 = stablehlo.add %1, %2 : tensor<4x8xf32> // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = {{.*}}"m1", "m2">}
  %5 = stablehlo.add %4, %3 : tensor<4x8xf32> // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = {{.*}}"m1", "m2", "m3">}
  func.return %5 : tensor<4x8xf32>
}

// CHECK-LABEL: func @reduce_of_reduces
func.func @reduce_of_reduces(%arg0: !m1_4x8, %arg1: !m2_4x8)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  %1 = mpmd.unassign %arg0 : (!m1_4x8) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  %2 = mpmd.unassign %arg1 : (!m2_4x8) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  %41 = stablehlo.add %1, %2 : tensor<4x8xf32> // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = {{.*}}"m1", "m2">}
  %42 = stablehlo.add %1, %2 : tensor<4x8xf32> // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = {{.*}}"m1", "m2">}
  %5 = stablehlo.add %41, %42 : tensor<4x8xf32> // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = {{.*}}"m1", "m2">}
  func.return %5 : tensor<4x8xf32>
}

// CHECK-LABEL: func @different_reduce_types
func.func @different_reduce_types(%arg0: !m1_4x8, %arg1: !m2_4x8)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  %1 = mpmd.unassign %arg0 : (!m1_4x8) -> tensor<4x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  %2 = mpmd.unassign %arg1 : (!m2_4x8) -> tensor<4x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  %41 = stablehlo.add %1, %2 : tensor<4x8xf32> // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = {{.*}}"m1", "m2">}
  %42 = stablehlo.add %1, %2 : tensor<4x8xf32> // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = {{.*}}"m1", "m2">}
  %5 = stablehlo.maximum %41, %42 : tensor<4x8xf32> // CHECK-NEXT: {mpmd.src_set = {{.*}}"m1", "m2">}
  func.return %5 : tensor<4x8xf32>
}

// CHECK-LABEL: func @concat_reduce(%arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m2"
func.func @concat_reduce(%arg0: !m1_4x1x8, %arg1: !m2_4x1x8)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  %init = stablehlo.constant dense<1.0> : tensor<f32>
  %1 = mpmd.unassign %arg0 : (!m1_4x1x8) -> tensor<4x1x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  %2 = mpmd.unassign %arg1 : (!m2_4x1x8) -> tensor<4x1x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  // The src_sets would normally be empty, but we parse the concat-reduce pair.
  // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = {{.*}}"m1", "m2">}
  %concat = "stablehlo.concatenate"(%1, %2, %1, %2) <{dimension = 1 : i64}> :
    (tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>) -> tensor<4x4x8xf32>
  // CHECK-NEXT: {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = {{.*}}"m1", "m2">}
  %reduce = stablehlo.reduce(%concat init: %init) applies stablehlo.maximum across dimensions = [1] :
    (tensor<4x4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  func.return %reduce : tensor<4x8xf32>
}

// CHECK-LABEL: func @cannot_parse_concat_reduce_unless_all_collapsed_dim_size_1
func.func @cannot_parse_concat_reduce_unless_all_collapsed_dim_size_1(%arg0: !m1_4x1x8, %arg1: !m2_4x1x8)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  %init = stablehlo.constant dense<1.0> : tensor<f32>
  %1 = mpmd.unassign %arg0 : (!m1_4x1x8) -> tensor<4x1x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  %2 = mpmd.unassign %arg1 : (!m2_4x1x8) -> tensor<4x1x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  // CHECK-NEXT: src_set = {{.*}}>
  %concat0 = "stablehlo.concatenate"(%1, %2) <{dimension = 1 : i64}> :
    (tensor<4x1x8xf32>, tensor<4x1x8xf32>) -> tensor<4x2x8xf32>
  // CHECK-NEXT: src_set = {{.*}}>
  %concat = "stablehlo.concatenate"(%concat0, %1, %2) <{dimension = 1 : i64}> :
    (tensor<4x2x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>) -> tensor<4x4x8xf32>
  // CHECK-NEXT: src_set = {{.*}}>
  %reduce = stablehlo.reduce(%concat init: %init) applies stablehlo.maximum across dimensions = [1] :
    (tensor<4x4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  func.return %reduce : tensor<4x8xf32>
}

// CHECK-LABEL: func @cannot_parse_concat_reduce_if_concat_multiple_users
func.func @cannot_parse_concat_reduce_if_concat_multiple_users(%arg0: !m1_4x1x8, %arg1: !m2_4x1x8)
  -> (tensor<4x8xf32>, tensor<4x4x8xf32>) attributes {topology=#topology} {
  %init = stablehlo.constant dense<1.0> : tensor<f32>
  %1 = mpmd.unassign %arg0 : (!m1_4x1x8) -> tensor<4x1x8xf32>  // CHECK: unassign {{.*}}src_set = {{.*}}"m1">
  %2 = mpmd.unassign %arg1 : (!m2_4x1x8) -> tensor<4x1x8xf32>  // CHECK-NEXT: src_set = {{.*}}"m2">
  // The src_sets would normally be empty, but we parse the concat-reduce pair.
  // CHECK-NEXT: src_set = {{.*}}>
  %concat = "stablehlo.concatenate"(%1, %2, %1, %2) <{dimension = 1 : i64}> :
    (tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>) -> tensor<4x4x8xf32>
  // CHECK-NEXT: src_set = {{.*}}>
  %reduce = stablehlo.reduce(%concat init: %init) applies stablehlo.maximum across dimensions = [1] :
    (tensor<4x4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  func.return %reduce, %concat : tensor<4x8xf32>, tensor<4x4x8xf32>
}
