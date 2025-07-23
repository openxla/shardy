// RUN: mpmd_opt %s -mpmd-infer-mesh-rewrite-using-analysis 2>&1 | FileCheck %s

!mesh_1_tensor_ui32 = !mpmd.mesh_tensor<"m1", tensor<ui32>>
!mesh_1_tensor_f32 = !mpmd.mesh_tensor<"m1", tensor<f32>>
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_ui32 = !mpmd.mesh_tensor<"m1", tensor<ui32>>
!mesh_2_tensor_f32 = !mpmd.mesh_tensor<"m2", tensor<f32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
!mesh_3_tensor_4_8_f32 = !mpmd.mesh_tensor<"m3", tensor<4x8xf32>>

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>>

// CHECK-LABEL: func @simple_rewrite(%arg0: tensor<4x8xf32>
func.func @simple_rewrite(%arg0: tensor<4x8xf32> {mpmd.src_set = #mpmd.meshes_with_origins<"m1">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">}) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // CHECK-NEXT:  %[[ASSIGN:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[ADD_FRAG:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ASSIGN]]) (%arg1: tensor<4x8xf32>) {
  // CHECK-NEXT:    %[[CONST:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg1, %[[CONST]]
  // CHECK-NEXT:    mpmd.return %[[ADD]]
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %[[UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG]]
  // CHECK-NEXT:  %[[RET_ASSIGN:.*]] = mpmd.assign {mpmd.use_set = {{.*}}"m1">} %[[UNASSIGN]]
  // CHECK-NEXT:  return %[[RET_ASSIGN]] : !mpmd.mesh_tensor<"m1"
  %0 = stablehlo.constant {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} dense<1.0> : tensor<4x8xf32>
  %2 = stablehlo.add %arg0, %0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
  %3 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %2 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  func.return %3 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @rewrite_with_duplication(%arg0: tensor<4x8xf32>
func.func @rewrite_with_duplication(%arg0: tensor<4x8xf32> {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">, mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {

  // CHECK-DAG:  %[[ASSIGN_ARG0_1:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1",

  // CHECK-DAG:   %[[ADD_FRAG_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ASSIGN_ARG0_1]]) (%arg1: tensor<4x8xf32>) {
  // CHECK-NEXT:    %[[CONST_1:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-NEXT:    %[[ADD_1:.*]] = stablehlo.add %arg1, %[[CONST_1]]
  // CHECK-NEXT:    mpmd.return %[[ADD_1]]
  // CHECK-NEXT:  }
  // CHECK-DAG:   %[[UNASSIGN_1:.*]] = mpmd.unassign %[[ADD_FRAG_1]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-DAG:   %[[ASSIGN_ARG0_2:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2",
  // CHECK-DAG:   %[[ADD_FRAG_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[ASSIGN_ARG0_2]]) (%arg1: tensor<4x8xf32>) {
  // CHECK-NEXT:    %[[CONST_2:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-NEXT:    %[[ADD_2:.*]] = stablehlo.add %arg1, %[[CONST_2]]
  // CHECK-NEXT:    mpmd.return %[[ADD_2]]
  // CHECK-NEXT:  }
  // CHECK-DAG:   %[[UNASSIGN_2:.*]] = mpmd.unassign %[[ADD_FRAG_2]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-DAG:   %[[ASSIGN_1:.*]] = mpmd.assign {mpmd.use_set = {{.*}}"m1">} %[[UNASSIGN_1]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-DAG:   %[[ASSIGN_2:.*]] = mpmd.assign {mpmd.use_set = {{.*}}"m2">} %[[UNASSIGN_2]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-DAG:   return %[[ASSIGN_1]], %[[ASSIGN_2]] :
  // CHECK-SAME:     !mpmd.mesh_tensor<"m1"
  // CHECK-SAME:     !mpmd.mesh_tensor<"m2"
  %0 = stablehlo.constant {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} dense<1.0> : tensor<4x8xf32>
  %2 = stablehlo.add %arg0, %0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %3 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %2 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %5 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %2 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  func.return %3, %5 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @call_op
func.func @call_op(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  // CHECK-NEXT:  %[[ARG0_ASSIGN:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[ADD_FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ARG0_ASSIGN]]) (%arg2
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-NEXT:    mpmd.return %[[ADD]]
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %[[F0_UNASSIGN:.*]] = mpmd.unassign %1 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-NEXT:  %[[ARG1_ASSIGN:.*]] = mpmd.assign %arg1 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[ADD_FRAG1:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[ARG1_ASSIGN]]) (%arg2
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-NEXT:    mpmd.return %[[ADD]]
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %[[F1_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG1]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-NEXT:  %[[F0_ASSIGN:.*]] = mpmd.assign %[[F0_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[F1_ASSIGN:.*]] = mpmd.assign %[[F1_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // We verify that the call op is replaced but the call_counter is kept.
  // CHECK-NEXT:  %[[CALL:.*]]:2 = mpmd.call @call_op_f(%[[F0_ASSIGN]], %[[F1_ASSIGN]])
  // CHECK-SAME:         {call_counter = 0 : ui32}

  // CHECK-NEXT: mpmd.unassign %[[CALL]]#0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-NEXT: mpmd.unassign %[[CALL]]#1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>

  %2:2 = mpmd.call @call_op_f(%0, %1) {call_counter = 0 : ui32}
    : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %6 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %2#0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %7 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %2#1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  func.return %6, %7 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK: func.func private @call_op_f
// CHECK-SAME:   arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:   arg1: !mpmd.mesh_tensor<"m2"
func.func private @call_op_f(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">},
  %arg1: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2">}
) -> (
  tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">},
  tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2">})
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  // CHECK-DAG: %[[UNASSIGN_ARG0:.*]] = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-DAG: %[[UNASSIGN_ARG1:.*]] = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-DAG: %[[ASSIGN_ARG0:.*]] = mpmd.assign %[[UNASSIGN_ARG0]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-DAG: %[[ASSIGN_ARG1:.*]] = mpmd.assign %[[UNASSIGN_ARG1]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

  // CHECK-DAG:   %[[ADD_FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ASSIGN_ARG0]]) (%arg2
  // CHECK-DAG:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-DAG:    mpmd.return %[[ADD]]
  // CHECK-DAG:  }

  // CHECK-DAG:   %[[ADD_FRAG1:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[ASSIGN_ARG1]]) (%arg2
  // CHECK-DAG:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-DAG:    mpmd.return %[[ADD]]
  // CHECK-DAG:  }

  // CHECK-DAG: %[[F0_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG0]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-DAG: %[[F1_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG1]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-DAG: mpmd.assign %[[F0_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-DAG: mpmd.assign %[[F1_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>

  func.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}


// CHECK-LABEL: func @call_op_multiple_calls
func.func @call_op_multiple_calls(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  // CHECK-NEXT:  %[[ARG0_ASSIGN:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[ADD_FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ARG0_ASSIGN]]) (%arg2
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-NEXT:    mpmd.return %[[ADD]]
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %[[F0_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG0]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-NEXT:  %[[ARG1_ASSIGN:.*]] = mpmd.assign %arg1 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[ADD_FRAG1:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[ARG1_ASSIGN]]) (%arg2
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-NEXT:    mpmd.return %[[ADD]]
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %[[F1_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG1]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-NEXT:  %[[F0_ASSIGN:.*]] = mpmd.assign %[[F0_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[F1_ASSIGN:.*]] = mpmd.assign %[[F1_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[CALL0:.*]]:2 = mpmd.call @call_op_multiple_calls_f(%[[F0_ASSIGN]], %[[F1_ASSIGN]])
  // CHECK-NEXT:  %[[CALL0_UNASSIGN0:.*]] = mpmd.unassign %[[CALL0]]#0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-NEXT:  %[[CALL0_UNASSIGN1:.*]] = mpmd.unassign %[[CALL0]]#1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-NEXT:  %[[CALL0_ASSIGN0:.*]] = mpmd.assign {mpmd.use_set = {{.*}}"m1">} %[[CALL0_UNASSIGN0]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[CALL0_UNASSIGN0:.*]] = mpmd.unassign %[[CALL0_ASSIGN0]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-NEXT:  %[[CALL0_ASSIGN0:.*]] = mpmd.assign %[[CALL0_UNASSIGN0]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[CALL0_ASSIGN1:.*]] = mpmd.assign %[[CALL0_UNASSIGN1]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[CALL1:.*]]:2 = mpmd.call @call_op_multiple_calls_f(%[[CALL0_ASSIGN0]], %[[CALL0_ASSIGN1]])
  // CHECK-NEXT:  %[[CALL1_UNASSIGN0:.*]] = mpmd.unassign %[[CALL1]]#0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-NEXT:  %[[CALL1_UNASSIGN1:.*]] = mpmd.unassign %[[CALL1]]#1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-NEXT:  %[[CALL1_ASSIGN0:.*]] = mpmd.assign {mpmd.use_set = {{.*}}"m1">} %[[CALL1_UNASSIGN0]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[CALL1_ASSIGN1:.*]] = mpmd.assign {mpmd.use_set = {{.*}}"m2">} %[[CALL1_UNASSIGN1]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT:  return %[[CALL1_ASSIGN0]], %[[CALL1_ASSIGN1]]

  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>

  %2:2 = mpmd.call @call_op_multiple_calls_f(%0, %1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %3 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %2#0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %31 = mpmd.unassign %3 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>


  %4:2 = mpmd.call @call_op_multiple_calls_f(%31, %2#1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %6 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %4#0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %7 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %4#1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  func.return %6, %7 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}
// CHECK: func.func private @call_op_multiple_calls_f
// CHECK-SAME:   arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:   arg1: !mpmd.mesh_tensor<"m2"
func.func private @call_op_multiple_calls_f(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">},
  %arg1: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2">}
) -> (
  tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">},
  tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2">}
)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  // CHECK:       %[[ADD_FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]> ({{.*}}) (%arg2
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-NEXT:    mpmd.return %[[ADD]]
  // CHECK-NEXT:  }
  // CHECK:       %[[ADD_FRAG1:.*]] = mpmd.fragment<mesh="m2", origin=[]> ({{.*}}) (%arg2
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-NEXT:    mpmd.return %[[ADD]]
  // CHECK-NEXT:  }
  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>

  func.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @call_op_unused_output
func.func @call_op_unused_output(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  // CHECK-NEXT:  %[[ARG0_ASSIGN:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[ADD_FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ARG0_ASSIGN]]) (%arg2
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-NEXT:    mpmd.return %[[ADD]]
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %[[F0_UNASSIGN:.*]] = mpmd.unassign %1 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-NEXT:  %[[ARG1_ASSIGN:.*]] = mpmd.assign %arg1 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[ADD_FRAG1:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[ARG1_ASSIGN]]) (%arg2
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-NEXT:    mpmd.return %[[ADD]]
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %[[F1_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG1]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-NEXT:  %[[F0_ASSIGN:.*]] = mpmd.assign %[[F0_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[F1_ASSIGN:.*]] = mpmd.assign %[[F1_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT:  %[[CALL:.*]]:2 = mpmd.call @call_op_unused_output_f(%[[F0_ASSIGN]], %[[F1_ASSIGN]])

  // CHECK-NEXT: mpmd.unassign %[[CALL]]#0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>

  %2:2 = mpmd.call @call_op_unused_output_f(%0, %1) {call_counter = 0 : ui32}
    : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %6 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %2#0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  func.return %6, %6 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK: func.func private @call_op_unused_output_f
// CHECK-SAME:   arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:   arg1: !mpmd.mesh_tensor<"m2"
func.func private @call_op_unused_output_f(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">},
  %arg1: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2">}
) -> (
  tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">},
  tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2">}
)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  // CHECK-DAG: %[[UNASSIGN_ARG0:.*]] = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-DAG: %[[UNASSIGN_ARG1:.*]] = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-DAG: %[[ASSIGN_ARG0:.*]] = mpmd.assign %[[UNASSIGN_ARG0]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-DAG: %[[ASSIGN_ARG1:.*]] = mpmd.assign %[[UNASSIGN_ARG1]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

  // CHECK-DAG:   %[[ADD_FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ASSIGN_ARG0]]) (%arg2
  // CHECK-DAG:   %[[ADD_FRAG1:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[ASSIGN_ARG1]]) (%arg2

  // CHECK-DAG: %[[F0_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG0]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  // CHECK-DAG: %[[F1_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG1]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-DAG: mpmd.assign %[[F0_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-DAG: mpmd.assign %[[F1_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>

  func.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @rewrite_with_region_op_with_wrapping
func.func @rewrite_with_region_op_with_wrapping(%arg0: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // Given that there are no consumer fragments, the resulting two fragments
  //  result from wrapping the `while` op with fragments.

  // CHECK:        %[[WHILE_FRAG_1:.*]]:2 = mpmd.fragment<mesh="m1", origin=[]>
  // CHECK-DAG:      %[[CONST:.*]] = stablehlo.constant dense<1>
  // CHECK-DAG:      %[[WHILE:.*]]:2 = stablehlo.while
  // CHECK-DAG:      mpmd.return %[[WHILE]]#0, %[[WHILE]]#1
  // CHECK-NEXT:    }
  // CHECK:        %[[WHILE_FRAG_2:.*]]:2 = mpmd.fragment<mesh="m2", origin=[]>
  // CHECK-DAG:      %[[CONST:.*]] = stablehlo.constant dense<1>
  // CHECK-DAG:      %[[WHILE:.*]]:2 = stablehlo.while
  // CHECK-DAG:      mpmd.return %[[WHILE]]#0, %[[WHILE]]#1
  // CHECK-NEXT:    }
  %0 = stablehlo.constant {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} dense<1> : tensor<i32>
  %5:2 = stablehlo.while(%iterArg_0 = %0, %iterArg_1 = %arg0)  : tensor<i32>, tensor<4x8xf32>
   attributes {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}
   cond {
    %6 = "stablehlo.compare"(%iterArg_0, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%6) : (tensor<i1>) -> ()
  } do {
    %8 = stablehlo.add %iterArg_1, %iterArg_1 : tensor<4x8xf32>
    "stablehlo.return"(%iterArg_0, %8) : (tensor<i32>, tensor<4x8xf32>) -> ()
  }
  %6 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %5#1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %7 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %5#1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32

  func.return %6, %7 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @rewrite_with_region_op_wrap_because_no_cloning_allowed
func.func @rewrite_with_region_op_wrap_because_no_cloning_allowed(%arg0: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  // Although the while loop is consumed by fragments, it has multiple users and
  // therefore won't be absorbed into the consumers, as cloning is disabled.
  // Instead, it will also be wrapped in two freshly created fragments.

  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // CHECK:        %[[WHILE_FRAG_1:.*]]:2 = mpmd.fragment<mesh="m1", origin=[]>
  // CHECK-DAG:      %[[CONST:.*]] = stablehlo.constant dense<1>
  // CHECK-DAG:      %[[WHILE:.*]]:2 = stablehlo.while
  // CHECK-DAG:      mpmd.return %[[WHILE]]#0, %[[WHILE]]#1
  // CHECK-NEXT:    }
  // CHECK:        %[[WHILE_FRAG_2:.*]]:2 = mpmd.fragment<mesh="m2", origin=[]>
  // CHECK-DAG:      %[[CONST:.*]] = stablehlo.constant dense<1>
  // CHECK-DAG:      %[[WHILE:.*]]:2 = stablehlo.while
  // CHECK-DAG:      mpmd.return %[[WHILE]]#0, %[[WHILE]]#1
  // CHECK-NEXT:    }
  // CHECK:         mpmd.fragment<mesh="m1", origin=[]>
  // CHECK:         mpmd.fragment<mesh="m2", origin=[]>
  %0 = stablehlo.constant {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} dense<1> : tensor<i32>
  %5:2 = stablehlo.while(%iterArg_0 = %0, %iterArg_1 = %arg0)  : tensor<i32>, tensor<4x8xf32>
   attributes {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}
   cond {
    %6 = "stablehlo.compare"(%iterArg_0, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%6) : (tensor<i1>) -> ()
  } do {
    %8 = stablehlo.add %iterArg_1, %iterArg_1 : tensor<4x8xf32>
    "stablehlo.return"(%iterArg_0, %8) : (tensor<i32>, tensor<4x8xf32>) -> ()
  }
  %6 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %5#1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %8 = mpmd.fragment<mesh="m1", origin=[]> (%6) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %7 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %5#1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  %9 = mpmd.fragment<mesh="m2", origin=[]> (%7) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_2_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  func.return %8, %9 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @rewrite_with_region_op_via_inline
func.func @rewrite_with_region_op_via_inline(%arg0: tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  // The while loop has a single fragment consumer and therefore can be inlined.

  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT: %[[FRAG:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ASSIGN]])
  // CHECK-NEXT:    %[[CONST:.*]] = stablehlo.constant dense<1>
  // CHECK-NEXT:    %[[WHILE:.*]]:2 = stablehlo.while
  // Only the second result is used.
  // CHECK:         mpmd.return %[[WHILE]]#1
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[FRAG]]

  %0 = stablehlo.constant {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} dense<1> : tensor<i32>
  %5:2 = stablehlo.while(%iterArg_0 = %0, %iterArg_1 = %arg0)  : tensor<i32>, tensor<4x8xf32>
   attributes {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}
   cond {
    %6 = "stablehlo.compare"(%iterArg_0, %0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%6) : (tensor<i1>) -> ()
  } do {
    %8 = stablehlo.add %iterArg_1, %iterArg_1 : tensor<4x8xf32>
    "stablehlo.return"(%iterArg_0, %8) : (tensor<i32>, tensor<4x8xf32>) -> ()
  }
  %6 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %5#1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %8 = mpmd.fragment<mesh="m1", origin=[]> (%6) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %8 : !mesh_1_tensor_4_8_f32
}

// Caller expectations:
// 1. arg0 is used in two meshes, so it will be transferred.
// 2. The `multiply` op is assigned to two meshes. Therefore, we should see it
// cloned and in two different fragments. Each clone is used in a different call
// operand.
// 3. The `add` op needs no cloning.

// CHECK-LABEL: func @call_op_with_multi_result_assignment
func.func @call_op_with_multi_result_assignment(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_1_tensor_f32, !mesh_2_tensor_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{

  // CHECK-NEXT: %[[ARG0_ASSIGN_0:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT: %[[MULT_FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ARG0_ASSIGN_0]])
  // CHECK-NEXT:   stablehlo.multiply
  // CHECK:      %[[MULT_FRAG0_UNASSIGN:.*]] = mpmd.unassign %[[MULT_FRAG0]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-NEXT: %[[ARG0_ASSIGN_1:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT: %[[MULT_FRAG1:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[ARG0_ASSIGN_1]]) (%arg2: tensor<4x8xf32>) {
  // CHECK-NEXT:   stablehlo.multiply
  // CHECK:      %[[MULT_FRAG1_UNASSIGN:.*]] = mpmd.unassign %[[MULT_FRAG1]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-NEXT: %[[ARG1_ASSIGN:.*]] = mpmd.assign %arg1 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT: %[[ADD_FRAG:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[ARG1_ASSIGN]]) (%arg2: tensor<4x8xf32>) {
  // CHECK-NEXT:   stablehlo.add
  // CHECK:      %[[ADD_FRAG_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>

  // CHECK-NEXT: %[[MULT_FRAG0_ASSIGN:.*]] = mpmd.assign %[[MULT_FRAG0_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT: %[[MULT_FRAG1_ASSIGN:.*]] = mpmd.assign %[[MULT_FRAG1_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT: %[[ADD_FRAG_ASSIGN:.*]] = mpmd.assign %[[ADD_FRAG_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT: mpmd.call @call_op_g(%[[MULT_FRAG0_ASSIGN]], %[[ADD_FRAG_ASSIGN]], %[[MULT_FRAG1_ASSIGN]])

  %0 = stablehlo.multiply %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>

  %2:3 = mpmd.call @call_op_g(%0, %1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>)
  %a1 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %2#0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %a2 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %2#0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32

  %7 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %2#1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32

  %a3 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %2#2 : (tensor<f32>) -> !mesh_1_tensor_f32
  %a4 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %2#2 : (tensor<f32>) -> !mesh_2_tensor_f32
  func.return %a1, %a2, %7, %a3, %a4 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_1_tensor_f32, !mesh_2_tensor_f32
}

// Callee expectations:
// 1. The `multiply` op is assigned to two meshes. Therefore, we should see it
// cloned and in two different fragments. The result of each fragment is
// returned by the callee function, i.e., the pass will replicate function
// results to guarantee assignment to a single mesh of the op.
// 2. The `add` op is assigned to a single mesh. Therefore, we should see it
// in a single fragment.
// 3. The `constant` op is assigned to two meshes. Similar behaviour to (1).
// 4. %arg0 is replicated to guarantee every argument is assigned to a single
// mesh.

// CHECK: func.func private @call_op_g
// CHECK-SAME:   arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:   arg1: !mpmd.mesh_tensor<"m2"
// CHECK-SAME:   arg2: !mpmd.mesh_tensor<"m2"
func.func private @call_op_g(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">},
  %arg1: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2">}
) -> (
  tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">},
  tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2">},
  tensor<f32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}
)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  // CHECK:       mpmd.fragment<mesh="m1", origin=[]>
  // CHECK-NEXT:    stablehlo.multiply
  // CHECK:       mpmd.fragment<mesh="m2", origin=[]>
  // CHECK-NEXT:    stablehlo.multiply

  // CHECK:       mpmd.fragment<mesh="m2", origin=[]>
  // CHECK-NEXT:    stablehlo.add

  // CHECK:       mpmd.fragment<mesh="m1", origin=[]>
  // CHECK-NEXT:    stablehlo.constant

  // CHECK:       mpmd.fragment<mesh="m2", origin=[]>
  // CHECK-NEXT:    stablehlo.constant

  // CHECK:       return {{.*}}m1{{.*}}m2{{.*}}m1{{.*}}m2{{.*}}m2
  %0 = stablehlo.multiply %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>
  %2 = stablehlo.constant  {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} dense<1.000000e+00> : tensor<f32>
  func.return %0, %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>
}

// CHECK-LABEL: func @call_op_noop_return
func.func @call_op_noop_return(%arg0: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
// CHECK-NEXT: %[[ARG0_ASSIGN_0:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-NEXT: %[[ADD_FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ARG0_ASSIGN_0]]) (%arg1
// CHECK-NEXT:   stablehlo.add %arg1, %arg1
// CHECK-NEXT:   mpmd.return
// CHECK-NEXT: }
// CHECK-NEXT: %[[F0_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG0]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>

// CHECK-NEXT: %[[ARG0_ASSIGN_1:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
// CHECK-NEXT: %[[ADD_FRAG1:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[ARG0_ASSIGN_1]]) (%arg1
// CHECK-NEXT:   stablehlo.add %arg1, %arg1
// CHECK-NEXT:   mpmd.return
// CHECK-NEXT: }
// CHECK-NEXT: %[[F1_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG1]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>

  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>

// CHECK-NEXT: %[[F0_ASSIGN:.*]] = mpmd.assign %[[F0_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-NEXT: %[[F1_ASSIGN:.*]] = mpmd.assign %[[F1_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
// CHECK-NEXT: %[[CALL:.*]]:2 = mpmd.call @call_op_noop_f(%[[F0_ASSIGN]], %[[F1_ASSIGN]])
// CHECK-DAG:  %[[CALL_0_UNASSIGN:.*]] = mpmd.unassign %[[CALL]]#0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG:  %[[CALL_1_UNASSIGN:.*]] = mpmd.unassign %[[CALL]]#1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>

  %3 = mpmd.call @call_op_noop_f(%0) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %6 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %3 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %7 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %3 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
// CHECK-DAG:  %[[CALL_0_ASSIGN:.*]] = mpmd.assign {mpmd.use_set = {{.*}}"m1">} %[[CALL_0_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-DAG:  %[[CALL_1_ASSIGN:.*]] = mpmd.assign {mpmd.use_set = {{.*}}"m2">} %[[CALL_1_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
// CHECK:      return %[[CALL_0_ASSIGN]], %[[CALL_1_ASSIGN]]
  func.return %6, %7 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}
// CHECK-LABEL: func private @call_op_noop_f
// CHECK-SAME:   arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:   arg1: !mpmd.mesh_tensor<"m2"
func.func private @call_op_noop_f(%arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}
) -> (tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">})
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
// CHECK-DAG: %[[ARG0_UNASSIGN:.*]] = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG: %[[ARG1_UNASSIGN:.*]] = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG: %[[ARG0_ASSIGN:.*]] = mpmd.assign %[[ARG0_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-DAG: %[[ARG1_ASSIGN:.*]] = mpmd.assign %[[ARG1_UNASSIGN]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
// CHECK-NEXT: return %[[ARG0_ASSIGN]], %[[ARG1_ASSIGN]]
  func.return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @call_op_chained_calls
func.func @call_op_chained_calls(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
// Each of these `add` operators wil be replicated in both meshes. This means
// we will have four fragments, two per mesh.
// CHECK: mpmd.fragment<mesh="m1"
// CHECK: mpmd.fragment<mesh="m2"
// CHECK: mpmd.fragment<mesh="m1"
// CHECK: mpmd.fragment<mesh="m2"

  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>

// CHECK:      %[[CALL0:.*]]:4 = mpmd.call @call_op_chained_calls_f
// CHECK-SAME:   {call_counter = 0 : ui32}
// CHECK-SAME:   (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>)

// CHECK-DAG:  %[[CALL0_UNASSIGN2:.*]] = mpmd.unassign %[[CALL0]]#2 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG:  %[[CALL0_ASSIGN2:.*]] = mpmd.assign %[[CALL0_UNASSIGN2]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-DAG:  %[[CALL0_UNASSIGN0:.*]] = mpmd.unassign %[[CALL0]]#0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG:  %[[CALL0_ASSIGN0:.*]] = mpmd.assign %[[CALL0_UNASSIGN0]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-DAG:  %[[CALL0_UNASSIGN3:.*]] = mpmd.unassign %[[CALL0]]#3 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG:  %[[CALL0_ASSIGN3:.*]] = mpmd.assign %[[CALL0_UNASSIGN3]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-DAG:  %[[CALL0_UNASSIGN1:.*]] = mpmd.unassign %[[CALL0]]#1 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG:  %[[CALL0_ASSIGN1:.*]] = mpmd.assign %[[CALL0_UNASSIGN1]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-NEXT: %[[CALL1:.*]]:4 = mpmd.call @call_op_chained_calls_f(%[[CALL0_ASSIGN0]], %[[CALL0_ASSIGN1]], %[[CALL0_ASSIGN2]], %[[CALL0_ASSIGN3]])
// CHECK-SAME:     {call_counter = 1 : ui32}
// CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>)
// CHECK-DAG:  %[[CALL1_UNASSIGN2:.*]] = mpmd.unassign %[[CALL1]]#2 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG:  %[[CALL1_UNASSIGN0:.*]] = mpmd.unassign %[[CALL1]]#0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG:  %[[CALL1_UNASSIGN3:.*]] = mpmd.unassign %[[CALL1]]#3 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG:  %[[CALL1_UNASSIGN1:.*]] = mpmd.unassign %[[CALL1]]#1 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>

  %2:2 = mpmd.call @call_op_chained_calls_f(%0, %1) {call_counter = 0 : ui32}
    : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %3:2 = mpmd.call @call_op_chained_calls_f(%2#0, %2#1) {call_counter = 1 : ui32}
    : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %7 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %3#0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %72 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %3#0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  %8 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %3#1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %82 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} %3#1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  func.return %7, %72, %8, %82 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}
// CHECK: func.func private @call_op_chained_calls_f
// CHECK-SAME:   arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:   arg1: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:   arg2: !mpmd.mesh_tensor<"m2"
// CHECK-SAME:   arg3: !mpmd.mesh_tensor<"m2"
func.func private @call_op_chained_calls_f(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">},
  %arg1: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}
) -> (
    tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">},
    tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}
)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
// CHECK: mpmd.fragment<mesh="m1"
// CHECK: mpmd.fragment<mesh="m2"
// CHECK: mpmd.fragment<mesh="m1"
// CHECK: mpmd.fragment<mesh="m2"
  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>

  func.return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @multiple_use_by_single_fragment_via_single_assign(%arg0: tensor<4x8xf32>
func.func @multiple_use_by_single_fragment_via_single_assign(%arg0: tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // Although the fragment has two uses of the same value, the resulting
  //  fragment has a single operand.
  // CHECK-NEXT: %[[ARG0_ASSIGN:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ARG0_ASSIGN]])
  // CHECK-SAME:   (%arg1: tensor<4x8xf32>)
  // CHECK-NEXT:   add
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[FRAGMENT]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  %1 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.fragment<mesh="m1", origin=[]> (%1, %1) (%arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg1, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @multiple_use_by_single_fragment_via_many_assign(%arg0: tensor<4x8xf32>
func.func @multiple_use_by_single_fragment_via_many_assign(%arg0: tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // Although the fragment has two uses of the same value, via different
  // assigns, the resulting fragment has a single operand.
  // CHECK-NEXT: %[[ARG0_ASSIGN:.*]] = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ARG0_ASSIGN]])
  // CHECK-SAME:   (%arg1: tensor<4x8xf32>)
  // CHECK-NEXT:   add
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[FRAGMENT]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  %a1 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %a2 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.fragment<mesh="m1", origin=[]> (%a1, %a2) (%arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg1, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @multiple_fragment_users_but_op_is_constant(
func.func @multiple_fragment_users_but_op_is_constant(%arg0: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // Although the constant has multiple fragment consumers, it is still cloned
  // into each one as it has no operands.
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=[]>
  // CHECK-NEXT:   constant
  // CHECK:      mpmd.fragment<mesh="m1", origin=[]>
  // CHECK-NEXT:   constant
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %1 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.fragment<mesh="m1", origin=[]> (%1) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %3 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %4 = mpmd.fragment<mesh="m1", origin=[]> (%3) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %2, %4 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @single_fragment_consumer_but_used_by_return(%arg0: tensor<4x8xf32>
func.func @single_fragment_consumer_but_used_by_return(%arg0: tensor<4x8xf32>) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // The constant cannot be inlined into the consumer fragment because it is
  // used by the return statement.
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=[]> () ()
  // CHECK-NEXT:   constant
  // CHECK:      mpmd.fragment<mesh="m1", origin=[]>
  %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %1 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.fragment<mesh="m1", origin=[]> (%1) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %3 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  func.return %2, %3 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @many_user_assigns_but_two_meshes
func.func @many_user_assigns_but_two_meshes(%arg0: tensor<8x16xi32>)
  -> (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>,
      !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>,
      !mpmd.mesh_tensor<"mesh2", tensor<8x16xi32>>,
      !mpmd.mesh_tensor<"mesh2", tensor<8x16xi32>>)
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>}
{
// CHECK:      mpmd.fragment<mesh="mesh1", origin=[]>
// CHECK-NEXT:   stablehlo.multiply %arg1, %arg1
// CHECK:      mpmd.fragment<mesh="mesh2", origin=[]>
// CHECK-NEXT:   stablehlo.multiply %arg1, %arg1

  %m = stablehlo.multiply %arg0, %arg0 : tensor<8x16xi32>
  %a1 = mpmd.assign %m : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %a2 = mpmd.assign %m : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %a3 = mpmd.assign %m : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh2", tensor<8x16xi32>>
  %a4 = mpmd.assign %m : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh2", tensor<8x16xi32>>
  return %a1, %a2, %a3, %a4 : !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh2", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh2", tensor<8x16xi32>>
}

// Illustrates a scenario where we end up with the same value (%arg0) passed to
// a fragment twice.
// CHECK-LABEL: func @op_operands_used_by_consumer
func.func @op_operands_used_by_consumer(%arg0: !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
// CHECK-NEXT: %[[ARG0_UNASSIGN:.*]] = mpmd.unassign %arg0
// CHECK-NEXT: %[[ARG0_ASSIGN:.*]] = mpmd.assign %[[ARG0_UNASSIGN]]
// CHECK-NEXT: mpmd.fragment<mesh="m1", origin=[]> (%arg0, %[[ARG0_ASSIGN]])
// CHECK-NEXT:   add
// CHECK-NEXT:   multiply
  %u = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %1 = stablehlo.add %u, %u : tensor<4x8xf32>
  %a = mpmd.assign %1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.fragment<mesh="m1", origin=[]> (%a, %arg0) (%arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg1, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @fori_loop
func.func @fori_loop(%arg0: tensor<ui32> {mpmd.src_set = #mpmd.meshes_with_origins<"m1">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">}) -> (tensor<ui32>, tensor<ui32>)
attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["y"=2]>>>} {
  // CHECK-NEXT:  %[[ARG0_ASSIGN:.*]] = mpmd.assign %arg0 : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:  %[[ADD_FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ARG0_ASSIGN]]) (%arg1: tensor<ui32>) {
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<ui32>
  // CHECK-NEXT:    mpmd.return %[[ADD]] : tensor<ui32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %[[F0_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG0]] : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>

  // CHECK-DAG:   %[[F0_ASSIGN:.*]] = mpmd.assign %[[F0_UNASSIGN]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-DAG:   %[[F1_ASSIGN:.*]] = mpmd.assign %[[F0_UNASSIGN]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:  %[[FOR:.*]]:2 = mpmd.for (%[[F0_ASSIGN]], %[[F1_ASSIGN]]) {iterations = 3 : ui32, unroll_factor = 3 : ui32} (
  // CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1", tensor<ui32>>,
  // CHECK-SAME:    %arg2: !mpmd.mesh_tensor<"m1", tensor<ui32>>,
  // CHECK-SAME:    %index: !mpmd.mesh_tensor<"m1", tensor<ui32>>) {
  // CHECK-DAG:     %[[UNASSIGN_ARG1:.*]] = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>
  // CHECK-DAG:     %[[UNASSIGN_ARG2:.*]] = mpmd.unassign %arg2 : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>
  // CHECK-DAG:     %[[UNASSIGN_INDEX:.*]] = mpmd.unassign %index : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>

  // CHECK-DAG:     %[[ASSIGN_INDEX:.*]] = mpmd.assign %[[UNASSIGN_INDEX]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-DAG:     %[[ASSIGN_ARG1:.*]] = mpmd.assign %[[UNASSIGN_ARG1]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-DAG:       %[[ADD_FRAG1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ASSIGN_INDEX]], %[[ASSIGN_ARG1]]) (%arg3: tensor<ui32>, %arg4: tensor<ui32>) {
  // CHECK-NEXT:      %[[CONST:.*]] = stablehlo.constant dense<1> : tensor<ui32>
  // CHECK-NEXT:      %[[ADD0:.*]] = stablehlo.add %arg4, %[[CONST]] : tensor<ui32>
  // CHECK-NEXT:      %[[ADD1:.*]] = stablehlo.add %[[ADD0]], %arg3 : tensor<ui32>
  // CHECK-NEXT:      mpmd.return %[[ADD1]] : tensor<ui32>
  // CHECK-NEXT:    } : (!mpmd.mesh_tensor<"m1", tensor<ui32>>, !mpmd.mesh_tensor<"m1", tensor<ui32>>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-DAG:     %[[F0_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG1]] : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>

  // CHECK-DAG:     %[[ASSIGN_ARG2:.*]] = mpmd.assign %[[UNASSIGN_ARG2]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-DAG:       %[[ADD_FRAG2:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ASSIGN_ARG2]]) (%arg3: tensor<ui32>) {
  // CHECK-NEXT:      %[[CONST:.*]] = stablehlo.constant dense<1> : tensor<ui32>
  // CHECK-NEXT:      %[[ADD0:.*]] = stablehlo.add %arg3, %[[CONST]] : tensor<ui32>
  // CHECK-NEXT:      mpmd.return %[[ADD0]] : tensor<ui32>
  // CHECK-NEXT:    } : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-DAG:     %[[F1_UNASSIGN:.*]] = mpmd.unassign %[[ADD_FRAG2]] : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>

  // CHECK-DAG:     %[[F0_ASSIGN:.*]] = mpmd.assign %[[F0_UNASSIGN]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-DAG:     %[[F1_ASSIGN:.*]] = mpmd.assign %[[F1_UNASSIGN]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:    mpmd.return %[[F0_ASSIGN]], %[[F1_ASSIGN]] : !mpmd.mesh_tensor<"m1", tensor<ui32>>, !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:  } : !mpmd.mesh_tensor<"m1", tensor<ui32>>, !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:  return %[[FOR]]#0, %[[FOR]]#1 : !mpmd.mesh_tensor<"m1", tensor<ui32>>, !mpmd.mesh_tensor<"m1", tensor<ui32>>

  %1 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m1">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<ui32>
  %2:2 = mpmd.for (%1, %1) {arg_attrs = [{mpmd.src_set = #mpmd.meshes_with_origins<"m1">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">}, {mpmd.src_set = #mpmd.meshes_with_origins<"m1">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">}, {mpmd.use_set = #mpmd.meshes_with_origins<"m1">}], iterations = 3 : ui32, unroll_factor = 3 : ui32} (%arg1: tensor<ui32>, %arg2: tensor<ui32>, %index: tensor<ui32>) {
    %3 = stablehlo.constant {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} dense<1> : tensor<ui32>
    %4 = stablehlo.add %arg1, %3 {mpmd.src_set = #mpmd.meshes_with_origins<"m1">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<ui32>
    %5 = stablehlo.add %4, %index {mpmd.src_set = #mpmd.meshes_with_origins<"m1">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<ui32>
    %6 = stablehlo.add %arg2, %3 {mpmd.src_set = #mpmd.meshes_with_origins<"m1">, mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<ui32>
    mpmd.return %5, %6 : tensor<ui32>, tensor<ui32>
  } : tensor<ui32>, tensor<ui32>
  return %2#0, %2#1 : tensor<ui32>, tensor<ui32>
}

// CHECK-LABEL: func @fori_loop_with_call
func.func @fori_loop_with_call(%arg0: tensor<ui32>) -> (!mesh_1_tensor_ui32)
attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK-NEXT:  %[[ASSIGN_ARG0:.*]] = mpmd.assign %arg0 : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:  %[[FOR:.*]] = mpmd.for (%[[ASSIGN_ARG0]]) {iterations = 3 : ui32, unroll_factor = 3 : ui32} (%arg1: !mpmd.mesh_tensor<"m1", tensor<ui32>>, %index: tensor<ui32>) {
  // CHECK-NEXT:    %[[UNASSIGN_ARG1:.*]] = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>
  // CHECK-DAG:     %[[ASSIGN_ARG1_1:.*]] = mpmd.assign %[[UNASSIGN_ARG1]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-DAG:     %[[ASSIGN_ARG1_2:.*]] = mpmd.assign %[[UNASSIGN_ARG1]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:    %[[CALL:.*]] = mpmd.call @fori_loop_with_call_f(%[[ASSIGN_ARG1_1]], %[[ASSIGN_ARG1_2]]) {mpmd.use_set = {{.*}}"m1">} : (!mpmd.mesh_tensor<"m1", tensor<ui32>>, !mpmd.mesh_tensor<"m1", tensor<ui32>>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:    %[[UNASSIGN_CALL:.*]] = mpmd.unassign %[[CALL]] : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>
  // CHECK-NEXT:    %[[ASSIGN_CALL:.*]] = mpmd.assign %[[UNASSIGN_CALL]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:    mpmd.return %[[ASSIGN_CALL]] : !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:  } : !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:  %[[UNASSIGN_FOR:.*]] = mpmd.unassign %[[FOR]] : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>
  // CHECK-NEXT:  %[[ASSIGN_FOR:.*]] = mpmd.assign {mpmd.use_set = {{.*}}"m1">} %[[UNASSIGN_FOR]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:  return %[[ASSIGN_FOR]] : !mpmd.mesh_tensor<"m1", tensor<ui32>>

  %0 = mpmd.for (%arg0) {arg_attrs = [{}, {mpmd.use_set = #mpmd.meshes_with_origins<"m1">}], iterations = 3 : ui32, unroll_factor = 3 : ui32} (%arg1: tensor<ui32>, %index: tensor<ui32>) {
    %2 = mpmd.call @fori_loop_with_call_f(%arg1, %arg1) {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    mpmd.return %2 : tensor<ui32>
  } : tensor<ui32>
  %1 = mpmd.assign {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} %0 : (tensor<ui32>) -> !mesh_1_tensor_ui32
  return %1 : !mesh_1_tensor_ui32
}

// CHECK-LABEL: func private @fori_loop_with_call_f
func.func private @fori_loop_with_call_f(%arg0: tensor<ui32>, %arg1: tensor<ui32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">}) -> tensor<ui32>
attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>>} {
  // CHECK-DAG:   %[[UNASSIGN_ARG0:.*]] = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>
  // CHECK-DAG:   %[[UNASSIGN_ARG1:.*]] = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>
  // CHECK-DAG:   %[[ASSIGN_ARG0:.*]] = mpmd.assign %[[UNASSIGN_ARG0]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-DAG:   %[[ASSIGN_ARG1:.*]] = mpmd.assign %[[UNASSIGN_ARG1]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-DAG:   %[[ADD_FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ASSIGN_ARG0]], %[[ASSIGN_ARG1]]) (%arg2: tensor<ui32>, %arg3: tensor<ui32>) {
  // CHECK-NEXT:    %[[ADD0:.*]] = stablehlo.add %arg2, %arg3 : tensor<ui32>
  // CHECK-NEXT:    mpmd.return %[[ADD0]] : tensor<ui32>
  // CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m1", tensor<ui32>>, !mpmd.mesh_tensor<"m1", tensor<ui32>>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:  %[[UNASSIGN_F0:.*]] = mpmd.unassign %[[ADD_FRAG0]] : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> tensor<ui32>
  // CHECK-NEXT:  %[[ASSIGN_F0:.*]] = mpmd.assign %[[UNASSIGN_F0]] : (tensor<ui32>) -> !mpmd.mesh_tensor<"m1", tensor<ui32>>
  // CHECK-NEXT:  return %[[ASSIGN_F0]] : !mpmd.mesh_tensor<"m1", tensor<ui32>>

  %0 = stablehlo.add %arg0, %arg1 : tensor<ui32>
  return %0 : tensor<ui32>
}
