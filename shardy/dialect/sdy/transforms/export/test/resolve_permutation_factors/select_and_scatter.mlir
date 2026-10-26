// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s --check-prefixes=CHECK,HALO

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>
// CHECK: sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>
sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>

// CHECK-LABEL: func @select_and_scatter_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<6x8xf32>)
func.func @select_and_scatter_permutation(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<6x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<f32>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<f32>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xf32>, tensor<6x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xf32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<f32>
  // HALO-NEXT: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[NEG_INF]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xf32>, %arg3: tensor<3x4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN:.*]] = stablehlo.multiply %[[REM]], %[[C_SIN]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_WIN_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE:.*]] = stablehlo.multiply %[[FIRST_WIN]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_PAD_LOW]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW:.*]] = stablehlo.subtract %[[ID_PLUS_PAD]], %[[FIRST_WIN_MUL_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [2:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD1_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD1_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BD1_PRED]], %[[CP1]], %[[BD1_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD2_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD2_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BD2_PRED]], %[[CP2]], %[[BD2_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xf32>, tensor<4x4xf32>, tensor<2x4xf32>) -> tensor<8x4xf32>
  // HALO-NEXT:   %[[C_OP_BASE:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW:.*]] = stablehlo.subtract %[[C_OP_BASE]], %[[OUT_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[OP_OFFSET]], %[[C_ZERO1]], sizes = [6, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<6x4xf32>
  // HALO-NEXT:   %[[SLICE_L_SRC:.*]] = stablehlo.slice %arg3 [2:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[C_CMP3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP3:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD3_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[BD3_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP3]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BD3_PRED]], %[[CP3]], %[[BD3_VAL]] : tensor<1x4xi1>, tensor<1x4xf32>
  // HALO-NEXT:   %[[SLICE_R_SRC:.*]] = stablehlo.slice %arg3 [0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[C_CMP4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP4:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD4_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[BD4_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP4]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL4:.*]] = stablehlo.select %[[BD4_PRED]], %[[CP4]], %[[BD4_VAL]] : tensor<1x4xi1>, tensor<1x4xf32>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %arg3, %[[SEL4]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<1x4xf32>, tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<5x4xf32>
  // HALO-NEXT:   %[[C_SSRC:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC:.*]] = stablehlo.multiply %[[REM]], %[[C_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN:.*]] = stablehlo.subtract %[[FIRST_WIN]], %[[ID_MUL_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW:.*]] = stablehlo.add %[[C_SRC_BASE]], %[[REL_FIRST_WIN]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[SRC_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<5x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // HALO-NEXT:   %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[SLICE1]], %[[SLICE2]], %arg4) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[COMP3:.*]] = stablehlo.compare GT, %arg6, %arg7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // HALO-NEXT:     stablehlo.return %[[COMP3]] : tensor<i1>
  // HALO-NEXT:   }, {
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[ADD5:.*]] = stablehlo.add %arg6, %arg7 : tensor<f32>
  // HALO-NEXT:     stablehlo.return %[[ADD5]] : tensor<f32>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<6x4xf32>
  // HALO-NEXT:   %[[DS2:.*]] = stablehlo.dynamic_slice %[[SS]], %[[OUT_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // HALO-NEXT:   sdy.return %[[DS2]] : tensor<4x4xf32>
  // HALO-NEXT: } : (tensor<8x8xf32>, tensor<6x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<6x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_permutation_min
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<6x8xf32>)
func.func @select_and_scatter_permutation_min(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<6x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare LT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<f32>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<f32>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xf32>, tensor<6x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xf32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<f32>
  // HALO-NEXT: %[[POS_INF:.*]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[POS_INF]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xf32>, %arg3: tensor<3x4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN:.*]] = stablehlo.multiply %[[REM]], %[[C_SIN]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_WIN_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE:.*]] = stablehlo.multiply %[[FIRST_WIN]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_PAD_LOW]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW:.*]] = stablehlo.subtract %[[ID_PLUS_PAD]], %[[FIRST_WIN_MUL_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [2:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD1_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD1_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BD1_PRED]], %[[CP1]], %[[BD1_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD2_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD2_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BD2_PRED]], %[[CP2]], %[[BD2_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xf32>, tensor<4x4xf32>, tensor<2x4xf32>) -> tensor<8x4xf32>
  // HALO-NEXT:   %[[C_OP_BASE:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW:.*]] = stablehlo.subtract %[[C_OP_BASE]], %[[OUT_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[OP_OFFSET]], %[[C_ZERO1]], sizes = [6, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<6x4xf32>
  // HALO-NEXT:   %[[SLICE_L_SRC:.*]] = stablehlo.slice %arg3 [2:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[C_CMP3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP3:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD3_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[BD3_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP3]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BD3_PRED]], %[[CP3]], %[[BD3_VAL]] : tensor<1x4xi1>, tensor<1x4xf32>
  // HALO-NEXT:   %[[SLICE_R_SRC:.*]] = stablehlo.slice %arg3 [0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[C_CMP4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP4:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD4_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[BD4_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP4]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL4:.*]] = stablehlo.select %[[BD4_PRED]], %[[CP4]], %[[BD4_VAL]] : tensor<1x4xi1>, tensor<1x4xf32>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %arg3, %[[SEL4]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<1x4xf32>, tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<5x4xf32>
  // HALO-NEXT:   %[[C_SSRC:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC:.*]] = stablehlo.multiply %[[REM]], %[[C_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN:.*]] = stablehlo.subtract %[[FIRST_WIN]], %[[ID_MUL_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW:.*]] = stablehlo.add %[[C_SRC_BASE]], %[[REL_FIRST_WIN]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[SRC_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<5x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // HALO-NEXT:   %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[SLICE1]], %[[SLICE2]], %arg4) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[COMP3:.*]] = stablehlo.compare LT, %arg6, %arg7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // HALO-NEXT:     stablehlo.return %[[COMP3]] : tensor<i1>
  // HALO-NEXT:   }, {
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[ADD5:.*]] = stablehlo.add %arg6, %arg7 : tensor<f32>
  // HALO-NEXT:     stablehlo.return %[[ADD5]] : tensor<f32>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<6x4xf32>
  // HALO-NEXT:   %[[DS2:.*]] = stablehlo.dynamic_slice %[[SS]], %[[OUT_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // HALO-NEXT:   sdy.return %[[DS2]] : tensor<4x4xf32>
  // HALO-NEXT: } : (tensor<8x8xf32>, tensor<6x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.compare LT, %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<6x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_permutation_int
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32> {{.*}}, %[[ARG1:.*]]: tensor<6x8xi32>)
func.func @select_and_scatter_permutation_int(
    %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<6x8xi32>)
    -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %cst = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<i32>, %[[R_ARG3:.*]]: tensor<i32>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<i32>, %[[R_ARG3:.*]]: tensor<i32>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<i32>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xi32>, tensor<6x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xi32>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xi32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<i32>
  // HALO-NEXT: %[[INT_MIN:.*]] = stablehlo.constant dense<-2147483648> : tensor<i32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[INT_MIN]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xi32>, %arg3: tensor<3x4xi32>, %arg4: tensor<i32>, %arg5: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN:.*]] = stablehlo.multiply %[[REM]], %[[C_SIN]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_WIN_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE:.*]] = stablehlo.multiply %[[FIRST_WIN]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_PAD_LOW]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW:.*]] = stablehlo.subtract %[[ID_PLUS_PAD]], %[[FIRST_WIN_MUL_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [2:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xi32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<2x4xi32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[C_CMP1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD1_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[BD1_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BD1_PRED]], %[[CP1]], %[[BD1_VAL]] : tensor<2x4xi1>, tensor<2x4xi32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xi32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<2x4xi32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[C_CMP2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD2_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[BD2_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BD2_PRED]], %[[CP2]], %[[BD2_VAL]] : tensor<2x4xi1>, tensor<2x4xi32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xi32>, tensor<4x4xi32>, tensor<2x4xi32>) -> tensor<8x4xi32>
  // HALO-NEXT:   %[[C_OP_BASE:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW:.*]] = stablehlo.subtract %[[C_OP_BASE]], %[[OUT_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[OP_OFFSET]], %c_1, sizes = [6, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xi32>, tensor<i32>, tensor<i32>) -> tensor<6x4xi32>
  // HALO-NEXT:   %[[SLICE_L_SRC:.*]] = stablehlo.slice %arg3 [2:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xi32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x4xi32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[C_CMP3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP3:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD3_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[BD3_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP3]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BD3_PRED]], %[[CP3]], %[[BD3_VAL]] : tensor<1x4xi1>, tensor<1x4xi32>
  // HALO-NEXT:   %[[SLICE_R_SRC:.*]] = stablehlo.slice %arg3 [0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xi32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x4xi32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[C_CMP4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP4:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD4_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[BD4_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP4]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL4:.*]] = stablehlo.select %[[BD4_PRED]], %[[CP4]], %[[BD4_VAL]] : tensor<1x4xi1>, tensor<1x4xi32>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %arg3, %[[SEL4]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<1x4xi32>, tensor<3x4xi32>, tensor<1x4xi32>) -> tensor<5x4xi32>
  // HALO-NEXT:   %[[C_SSRC:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC:.*]] = stablehlo.multiply %[[REM]], %[[C_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN:.*]] = stablehlo.subtract %[[FIRST_WIN]], %[[ID_MUL_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW:.*]] = stablehlo.add %[[C_SRC_BASE]], %[[REL_FIRST_WIN]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[SRC_OFFSET]], %c_1, sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<5x4xi32>, tensor<i32>, tensor<i32>) -> tensor<4x4xi32>
  // HALO-NEXT:   %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[SLICE1]], %[[SLICE2]], %arg4) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<i32>, %arg7: tensor<i32>):
  // HALO-NEXT:     %[[COMP3:.*]] = stablehlo.compare GT, %arg6, %arg7 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:     stablehlo.return %[[COMP3]] : tensor<i1>
  // HALO-NEXT:   }, {
  // HALO-NEXT:   ^bb0(%arg6: tensor<i32>, %arg7: tensor<i32>):
  // HALO-NEXT:     %[[ADD5:.*]] = stablehlo.add %arg6, %arg7 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD5]] : tensor<i32>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xi32>, tensor<4x4xi32>, tensor<i32>) -> tensor<6x4xi32>
  // HALO-NEXT:   %[[DS2:.*]] = stablehlo.dynamic_slice %[[SS]], %[[OUT_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xi32>, tensor<i32>, tensor<i32>) -> tensor<4x4xi32>
  // HALO-NEXT:   sdy.return %[[DS2]] : tensor<4x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<6x8xi32>, tensor<i32>, tensor<i32>) -> tensor<8x8xi32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<6x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// CHECK-LABEL: func @select_and_scatter_permutation_f64
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf64> {{.*}}, %[[ARG1:.*]]: tensor<6x8xf64>)
func.func @select_and_scatter_permutation_f64(
    %arg0: tensor<8x8xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<6x8xf64>)
    -> (tensor<8x8xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf64>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f64>, %[[R_ARG3:.*]]: tensor<f64>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<f64>, tensor<f64>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f64>, %[[R_ARG3:.*]]: tensor<f64>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<f64>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<f64>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xf64>, tensor<6x8xf64>, tensor<f64>) -> tensor<8x8xf64>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf64>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xf64>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<f64>
  // HALO-NEXT: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[NEG_INF]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xf64>, %arg3: tensor<3x4xf64>, %arg4: tensor<f64>, %arg5: tensor<f64>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN:.*]] = stablehlo.multiply %[[REM]], %[[C_SIN]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_WIN_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE:.*]] = stablehlo.multiply %[[FIRST_WIN]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_PAD_LOW]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW:.*]] = stablehlo.subtract %[[ID_PLUS_PAD]], %[[FIRST_WIN_MUL_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [2:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf64>) -> tensor<2x4xf64>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<2x4xf64>) -> tensor<2x4xf64>
  // HALO-NEXT:   %[[C_CMP1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD1_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f64>) -> tensor<2x4xf64>
  // HALO-NEXT:   %[[BD1_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BD1_PRED]], %[[CP1]], %[[BD1_VAL]] : tensor<2x4xi1>, tensor<2x4xf64>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf64>) -> tensor<2x4xf64>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<2x4xf64>) -> tensor<2x4xf64>
  // HALO-NEXT:   %[[C_CMP2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD2_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f64>) -> tensor<2x4xf64>
  // HALO-NEXT:   %[[BD2_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BD2_PRED]], %[[CP2]], %[[BD2_VAL]] : tensor<2x4xi1>, tensor<2x4xf64>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xf64>, tensor<4x4xf64>, tensor<2x4xf64>) -> tensor<8x4xf64>
  // HALO-NEXT:   %[[C_OP_BASE:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW:.*]] = stablehlo.subtract %[[C_OP_BASE]], %[[OUT_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[OP_OFFSET]], %[[C_ZERO1]], sizes = [6, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xf64>, tensor<i32>, tensor<i32>) -> tensor<6x4xf64>
  // HALO-NEXT:   %[[SLICE_L_SRC:.*]] = stablehlo.slice %arg3 [2:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xf64>) -> tensor<1x4xf64>
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x4xf64>) -> tensor<1x4xf64>
  // HALO-NEXT:   %[[C_CMP3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP3:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD3_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f64>) -> tensor<1x4xf64>
  // HALO-NEXT:   %[[BD3_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP3]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BD3_PRED]], %[[CP3]], %[[BD3_VAL]] : tensor<1x4xi1>, tensor<1x4xf64>
  // HALO-NEXT:   %[[SLICE_R_SRC:.*]] = stablehlo.slice %arg3 [0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xf64>) -> tensor<1x4xf64>
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x4xf64>) -> tensor<1x4xf64>
  // HALO-NEXT:   %[[C_CMP4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP4:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD4_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f64>) -> tensor<1x4xf64>
  // HALO-NEXT:   %[[BD4_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP4]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL4:.*]] = stablehlo.select %[[BD4_PRED]], %[[CP4]], %[[BD4_VAL]] : tensor<1x4xi1>, tensor<1x4xf64>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %arg3, %[[SEL4]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<1x4xf64>, tensor<3x4xf64>, tensor<1x4xf64>) -> tensor<5x4xf64>
  // HALO-NEXT:   %[[C_SSRC:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC:.*]] = stablehlo.multiply %[[REM]], %[[C_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN:.*]] = stablehlo.subtract %[[FIRST_WIN]], %[[ID_MUL_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW:.*]] = stablehlo.add %[[C_SRC_BASE]], %[[REL_FIRST_WIN]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[SRC_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<5x4xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
  // HALO-NEXT:   %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[SLICE1]], %[[SLICE2]], %arg4) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<f64>, %arg7: tensor<f64>):
  // HALO-NEXT:     %[[COMP3:.*]] = stablehlo.compare GT, %arg6, %arg7 : (tensor<f64>, tensor<f64>) -> tensor<i1>
  // HALO-NEXT:     stablehlo.return %[[COMP3]] : tensor<i1>
  // HALO-NEXT:   }, {
  // HALO-NEXT:   ^bb0(%arg6: tensor<f64>, %arg7: tensor<f64>):
  // HALO-NEXT:     %[[ADD5:.*]] = stablehlo.add %arg6, %arg7 : tensor<f64>
  // HALO-NEXT:     stablehlo.return %[[ADD5]] : tensor<f64>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xf64>, tensor<4x4xf64>, tensor<f64>) -> tensor<6x4xf64>
  // HALO-NEXT:   %[[DS2:.*]] = stablehlo.dynamic_slice %[[SS]], %[[OUT_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xf64>, tensor<i32>, tensor<i32>) -> tensor<4x4xf64>
  // HALO-NEXT:   sdy.return %[[DS2]] : tensor<4x4xf64>
  // HALO-NEXT: } : (tensor<8x8xf64>, tensor<6x8xf64>, tensor<f64>, tensor<f64>) -> tensor<8x8xf64>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f64>
      stablehlo.return %2 : tensor<f64>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf64>, tensor<6x8xf64>, tensor<f64>) -> tensor<8x8xf64>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: func @select_and_scatter_stride_padding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<4x8xf32>)
func.func @select_and_scatter_stride_padding(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<4x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // REPL-SAME{LITERAL}: <{padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 2, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<f32>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<f32>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xf32>, tensor<4x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<4x8xf32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<f32>
  // HALO-NEXT: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[NEG_INF]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xf32>, %arg3: tensor<2x4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN:.*]] = stablehlo.multiply %[[REM]], %[[C_SIN]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_WIN_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE:.*]] = stablehlo.multiply %[[FIRST_WIN]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_PAD_LOW]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW:.*]] = stablehlo.subtract %[[ID_PLUS_PAD]], %[[FIRST_WIN_MUL_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [3:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[C_CMP1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD1_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[BD1_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BD1_PRED]], %[[CP1]], %[[BD1_VAL]] : tensor<1x4xi1>, tensor<1x4xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD2_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD2_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BD2_PRED]], %[[CP2]], %[[BD2_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<1x4xf32>, tensor<4x4xf32>, tensor<2x4xf32>) -> tensor<7x4xf32>
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg5, low = [1, 0], high = [0, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<7x4xf32>, tensor<f32>) -> tensor<8x4xf32>
  // HALO-NEXT:   %[[C_OP_BASE:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW:.*]] = stablehlo.subtract %[[C_OP_BASE]], %[[OUT_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[OP_OFFSET]], %[[C_ZERO1]], sizes = [7, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<7x4xf32>
  // HALO-NEXT:   %[[SLICE_R_SRC:.*]] = stablehlo.slice %arg3 [0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[C_CMP4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP4:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD4_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[BD4_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP4]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL4:.*]] = stablehlo.select %[[BD4_PRED]], %[[CP4]], %[[BD4_VAL]] : tensor<1x4xi1>, tensor<1x4xf32>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %arg3, %[[SEL4]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  // HALO-NEXT:   %[[C_SSRC:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC:.*]] = stablehlo.multiply %[[REM]], %[[C_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN:.*]] = stablehlo.subtract %[[FIRST_WIN]], %[[ID_MUL_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW:.*]] = stablehlo.add %[[C_SRC_BASE]], %[[REL_FIRST_WIN]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[SRC_OFFSET]], %[[C_ZERO1]], sizes = [3, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xf32>, tensor<i32>, tensor<i32>) -> tensor<3x4xf32>
  // HALO-NEXT:   %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[SLICE1]], %[[SLICE2]], %arg4) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 2, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[COMP3:.*]] = stablehlo.compare GT, %arg6, %arg7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // HALO-NEXT:     stablehlo.return %[[COMP3]] : tensor<i1>
  // HALO-NEXT:   }, {
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[ADD5:.*]] = stablehlo.add %arg6, %arg7 : tensor<f32>
  // HALO-NEXT:     stablehlo.return %[[ADD5]] : tensor<f32>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<7x4xf32>, tensor<3x4xf32>, tensor<f32>) -> tensor<7x4xf32>
  // HALO-NEXT:   %[[DS2:.*]] = stablehlo.dynamic_slice %[[SS]], %[[OUT_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<7x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // HALO-NEXT:   sdy.return %[[DS2]] : tensor<4x4xf32>
  // HALO-NEXT: } : (tensor<8x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 2, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<4x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_asymmetric_padding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<16x8xf32>)
func.func @select_and_scatter_asymmetric_padding(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<16x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // REPL-SAME{LITERAL}: <{padding = dense<[[4, 6], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<f32>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<f32>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xf32>, tensor<16x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<16x8xf32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<f32>
  // HALO-NEXT: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[NEG_INF]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xf32>, %arg3: tensor<8x4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN:.*]] = stablehlo.multiply %[[REM]], %[[C_SIN]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_WIN_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE:.*]] = stablehlo.multiply %[[FIRST_WIN]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_PAD_LOW]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW:.*]] = stablehlo.subtract %[[ID_PLUS_PAD]], %[[FIRST_WIN_MUL_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [2:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD1_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD1_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BD1_PRED]], %[[CP1]], %[[BD1_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD2_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD2_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BD2_PRED]], %[[CP2]], %[[BD2_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xf32>, tensor<4x4xf32>, tensor<2x4xf32>) -> tensor<8x4xf32>
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg5, low = [2, 0], high = [2, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xf32>, tensor<f32>) -> tensor<12x4xf32>
  // HALO-NEXT:   %[[C_OP_BASE:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW:.*]] = stablehlo.subtract %[[C_OP_BASE]], %[[OUT_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[OP_OFFSET]], %[[C_ZERO1]], sizes = [8, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<12x4xf32>, tensor<i32>, tensor<i32>) -> tensor<8x4xf32>
  // HALO-NEXT:   %[[SLICE_L_SRC:.*]] = stablehlo.slice %arg3 [6:8, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP3:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD3_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD3_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP3]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BD3_PRED]], %[[CP3]], %[[BD3_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %arg3, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xf32>, tensor<8x4xf32>) -> tensor<10x4xf32>
  // HALO-NEXT:   %[[C_SSRC:.*]] = stablehlo.constant dense<8> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC:.*]] = stablehlo.multiply %[[REM]], %[[C_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN:.*]] = stablehlo.subtract %[[FIRST_WIN]], %[[ID_MUL_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW:.*]] = stablehlo.add %[[C_SRC_BASE]], %[[REL_FIRST_WIN]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[SRC_OFFSET]], %[[C_ZERO1]], sizes = [6, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<10x4xf32>, tensor<i32>, tensor<i32>) -> tensor<6x4xf32>
  // HALO-NEXT:   %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[SLICE1]], %[[SLICE2]], %arg4) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[COMP3:.*]] = stablehlo.compare GT, %arg6, %arg7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // HALO-NEXT:     stablehlo.return %[[COMP3]] : tensor<i1>
  // HALO-NEXT:   }, {
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[ADD5:.*]] = stablehlo.add %arg6, %arg7 : tensor<f32>
  // HALO-NEXT:     stablehlo.return %[[ADD5]] : tensor<f32>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xf32>, tensor<6x4xf32>, tensor<f32>) -> tensor<8x4xf32>
  // HALO-NEXT:   %[[DS2:.*]] = stablehlo.dynamic_slice %[[SS]], %[[OUT_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // HALO-NEXT:   sdy.return %[[DS2]] : tensor<4x4xf32>
  // HALO-NEXT: } : (tensor<8x8xf32>, tensor<16x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[4, 6], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<16x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_2d_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<8x8xf32>)
func.func @select_and_scatter_2d_sharded(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]]) <{padding = dense<1> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<f32>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<f32>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x8xf32>, tensor<8x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<f32>
  // HALO-NEXT: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[NEG_INF]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN:.*]] = stablehlo.multiply %[[REM]], %[[C_SIN]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_WIN_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE:.*]] = stablehlo.multiply %[[FIRST_WIN]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_PAD_LOW]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW:.*]] = stablehlo.subtract %[[ID_PLUS_PAD]], %[[FIRST_WIN_MUL_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [2:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD1_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD1_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BD1_PRED]], %[[CP1]], %[[BD1_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD2_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD2_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BD2_PRED]], %[[CP2]], %[[BD2_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xf32>, tensor<4x4xf32>, tensor<2x4xf32>) -> tensor<8x4xf32>
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg5, low = [1, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xf32>, tensor<f32>) -> tensor<10x4xf32>
  // HALO-NEXT:   %[[C_OP_BASE:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW:.*]] = stablehlo.subtract %[[C_OP_BASE]], %[[OUT_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[OP_OFFSET]], %[[C_ZERO1]], sizes = [7, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<10x4xf32>, tensor<i32>, tensor<i32>) -> tensor<7x4xf32>
  // HALO-NEXT:   %[[SLICE_L_SRC:.*]] = stablehlo.slice %arg3 [3:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[C_CMP3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP3:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD3_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[BD3_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP3]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BD3_PRED]], %[[CP3]], %[[BD3_VAL]] : tensor<1x4xi1>, tensor<1x4xf32>
  // HALO-NEXT:   %[[SLICE_R_SRC:.*]] = stablehlo.slice %arg3 [0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[C_CMP4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP4:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD4_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[BD4_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP4]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL4:.*]] = stablehlo.select %[[BD4_PRED]], %[[CP4]], %[[BD4_VAL]] : tensor<1x4xi1>, tensor<1x4xf32>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %arg3, %[[SEL4]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<1x4xf32>, tensor<4x4xf32>, tensor<1x4xf32>) -> tensor<6x4xf32>
  // HALO-NEXT:   %[[C_SSRC:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC:.*]] = stablehlo.multiply %[[REM]], %[[C_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN:.*]] = stablehlo.subtract %[[FIRST_WIN]], %[[ID_MUL_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW:.*]] = stablehlo.add %[[C_SRC_BASE]], %[[REL_FIRST_WIN]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[SRC_OFFSET]], %[[C_ZERO1]], sizes = [5, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xf32>, tensor<i32>, tensor<i32>) -> tensor<5x4xf32>
  // HALO-NEXT:   %[[C_MOD_D1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[MOD_D1:.*]] = stablehlo.remainder %[[PID1]], %[[C_MOD_D1]] : tensor<ui32>
  // HALO-NEXT:   %[[REM_D1:.*]] = stablehlo.convert %[[MOD_D1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN_D1:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN_D1:.*]] = stablehlo.multiply %[[REM_D1]], %[[C_SIN_D1]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET_D1:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM_D1:.*]] = stablehlo.add %[[ID_MUL_SIN_D1]], %[[C_WIN_OFFSET_D1]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP_D1:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM_D1]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1_D1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_D1:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP_D1]], %[[C_STRIDE1_D1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2_D1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE_D1:.*]] = stablehlo.multiply %[[FIRST_WIN_D1]], %[[C_STRIDE2_D1]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW_D1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD_D1:.*]] = stablehlo.add %[[ID_MUL_SIN_D1]], %[[C_PAD_LOW_D1]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW_D1:.*]] = stablehlo.subtract %[[ID_PLUS_PAD_D1]], %[[FIRST_WIN_MUL_STRIDE_D1]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_D1:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW_D1]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L_D1:.*]] = stablehlo.slice %[[SLICE1]] [0:7, 2:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<7x4xf32>) -> tensor<7x2xf32>
  // HALO-NEXT:   %[[CP5:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_D1]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : (tensor<7x2xf32>) -> tensor<7x2xf32>
  // HALO-NEXT:   %[[C_CMP5:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP5:.*]] = stablehlo.compare GE, %[[REM_D1]], %[[C_CMP5]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD5_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<7x2xf32>
  // HALO-NEXT:   %[[BD5_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP5]], dims = [] : (tensor<i1>) -> tensor<7x2xi1>
  // HALO-NEXT:   %[[SEL5:.*]] = stablehlo.select %[[BD5_PRED]], %[[CP5]], %[[BD5_VAL]] : tensor<7x2xi1>, tensor<7x2xf32>
  // HALO-NEXT:   %[[SLICE_R_D1:.*]] = stablehlo.slice %[[SLICE1]] [0:7, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<7x4xf32>) -> tensor<7x2xf32>
  // HALO-NEXT:   %[[CP6:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_D1]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}> : (tensor<7x2xf32>) -> tensor<7x2xf32>
  // HALO-NEXT:   %[[C_CMP6:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP6:.*]] = stablehlo.compare LT, %[[REM_D1]], %[[C_CMP6]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD6_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<7x2xf32>
  // HALO-NEXT:   %[[BD6_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP6]], dims = [] : (tensor<i1>) -> tensor<7x2xi1>
  // HALO-NEXT:   %[[SEL6:.*]] = stablehlo.select %[[BD6_PRED]], %[[CP6]], %[[BD6_VAL]] : tensor<7x2xi1>, tensor<7x2xf32>
  // HALO-NEXT:   %[[CONCAT3:.*]] = stablehlo.concatenate %[[SEL5]], %[[SLICE1]], %[[SEL6]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<7x2xf32>, tensor<7x4xf32>, tensor<7x2xf32>) -> tensor<7x8xf32>
  // HALO-NEXT:   %[[PAD3:.*]] = stablehlo.pad %[[CONCAT3]], %arg5, low = [0, 1], high = [0, 1], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<7x8xf32>, tensor<f32>) -> tensor<7x10xf32>
  // HALO-NEXT:   %[[C_OP_BASE_D1:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW_D1:.*]] = stablehlo.subtract %[[C_OP_BASE_D1]], %[[OUT_OFFSET_D1]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_D1:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW_D1]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE3:.*]] = stablehlo.dynamic_slice %[[PAD3]], %[[C_ZERO1]], %[[OP_OFFSET_D1]], sizes = [7, 7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<7x10xf32>, tensor<i32>, tensor<i32>) -> tensor<7x7xf32>
  // HALO-NEXT:   %[[SLICE_L_SRC_D1:.*]] = stablehlo.slice %[[SLICE2]] [0:5, 3:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<5x4xf32>) -> tensor<5x1xf32>
  // HALO-NEXT:   %[[CP7:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_SRC_D1]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : (tensor<5x1xf32>) -> tensor<5x1xf32>
  // HALO-NEXT:   %[[C_CMP7:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP7:.*]] = stablehlo.compare GE, %[[REM_D1]], %[[C_CMP7]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD7_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<5x1xf32>
  // HALO-NEXT:   %[[BD7_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP7]], dims = [] : (tensor<i1>) -> tensor<5x1xi1>
  // HALO-NEXT:   %[[SEL7:.*]] = stablehlo.select %[[BD7_PRED]], %[[CP7]], %[[BD7_VAL]] : tensor<5x1xi1>, tensor<5x1xf32>
  // HALO-NEXT:   %[[SLICE_R_SRC_D1:.*]] = stablehlo.slice %[[SLICE2]] [0:5, 0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<5x4xf32>) -> tensor<5x1xf32>
  // HALO-NEXT:   %[[CP8:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_SRC_D1]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}> : (tensor<5x1xf32>) -> tensor<5x1xf32>
  // HALO-NEXT:   %[[C_CMP8:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP8:.*]] = stablehlo.compare LT, %[[REM_D1]], %[[C_CMP8]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD8_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<5x1xf32>
  // HALO-NEXT:   %[[BD8_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP8]], dims = [] : (tensor<i1>) -> tensor<5x1xi1>
  // HALO-NEXT:   %[[SEL8:.*]] = stablehlo.select %[[BD8_PRED]], %[[CP8]], %[[BD8_VAL]] : tensor<5x1xi1>, tensor<5x1xf32>
  // HALO-NEXT:   %[[CONCAT4:.*]] = stablehlo.concatenate %[[SEL7]], %[[SLICE2]], %[[SEL8]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<5x1xf32>, tensor<5x4xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
  // HALO-NEXT:   %[[C_SSRC_D1:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC_D1:.*]] = stablehlo.multiply %[[REM_D1]], %[[C_SSRC_D1]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN_D1:.*]] = stablehlo.subtract %[[FIRST_WIN_D1]], %[[ID_MUL_SSRC_D1]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE_D1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW_D1:.*]] = stablehlo.add %[[C_SRC_BASE_D1]], %[[REL_FIRST_WIN_D1]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_D1:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW_D1]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE4:.*]] = stablehlo.dynamic_slice %[[CONCAT4]], %[[C_ZERO1]], %[[SRC_OFFSET_D1]], sizes = [5, 5] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<5x6xf32>, tensor<i32>, tensor<i32>) -> tensor<5x5xf32>
  // HALO-NEXT:   %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[SLICE3]], %[[SLICE4]], %arg4) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[COMP3:.*]] = stablehlo.compare GT, %arg6, %arg7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // HALO-NEXT:     stablehlo.return %[[COMP3]] : tensor<i1>
  // HALO-NEXT:   }, {
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[ADD5:.*]] = stablehlo.add %arg6, %arg7 : tensor<f32>
  // HALO-NEXT:     stablehlo.return %[[ADD5]] : tensor<f32>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<7x7xf32>, tensor<5x5xf32>, tensor<f32>) -> tensor<7x7xf32>
  // HALO-NEXT:   %[[DS2:.*]] = stablehlo.dynamic_slice %[[SS]], %[[OUT_OFFSET]], %[[OUT_OFFSET_D1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<7x7xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // HALO-NEXT:   sdy.return %[[DS2]] : tensor<4x4xf32>
  // HALO-NEXT: } : (tensor<8x8xf32>, tensor<8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 3>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<8x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_subaxis
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<6x8xf32>)
func.func @select_and_scatter_subaxis(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(2)2}, {"b"}]>},
    %arg1: tensor<6x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(2)2}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {"b"}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<f32>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<f32>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {"b"}]>]>} : (tensor<8x8xf32>, tensor<6x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh_a4, [{"a":(2)2}, {"b"}]> : tensor<8x8xf32>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh_a4, [{"a":(2)2}, {"b"}]> : tensor<6x8xf32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh_a4, []> : tensor<f32>
  // HALO-NEXT: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[NEG_INF]]) in_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>, <@mesh_a4, [{"a":(2)2}, {"b"}]>, <@mesh_a4, []>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>] manual_axes={"a":(2)2, "b"} (%arg2: tensor<4x4xf32>, %arg3: tensor<3x4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[MOD2:.*]] = stablehlo.remainder %[[DIV2]], %[[C_MOD2]] : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.convert %[[MOD2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN:.*]] = stablehlo.multiply %[[REM]], %[[C_SIN]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_WIN_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE:.*]] = stablehlo.multiply %[[FIRST_WIN]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_PAD_LOW]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW:.*]] = stablehlo.subtract %[[ID_PLUS_PAD]], %[[FIRST_WIN_MUL_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [2:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [4, 6], [5, 7]]> : tensor<4x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD1_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD1_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BD1_PRED]], %[[CP1]], %[[BD1_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [6, 4], [7, 5]]> : tensor<4x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[C_CMP2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD2_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[BD2_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BD2_PRED]], %[[CP2]], %[[BD2_VAL]] : tensor<2x4xi1>, tensor<2x4xf32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<2x4xf32>, tensor<4x4xf32>, tensor<2x4xf32>) -> tensor<8x4xf32>
  // HALO-NEXT:   %[[C_OP_BASE:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW:.*]] = stablehlo.subtract %[[C_OP_BASE]], %[[OUT_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[OP_OFFSET]], %[[C_ZERO1]], sizes = [6, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<6x4xf32>
  // HALO-NEXT:   %[[SLICE_L_SRC:.*]] = stablehlo.slice %arg3 [2:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<3x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [4, 6], [5, 7]]> : tensor<4x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[C_CMP3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP3:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD3_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[BD3_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP3]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BD3_PRED]], %[[CP3]], %[[BD3_VAL]] : tensor<1x4xi1>, tensor<1x4xf32>
  // HALO-NEXT:   %[[SLICE_R_SRC:.*]] = stablehlo.slice %arg3 [0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<3x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [6, 4], [7, 5]]> : tensor<4x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[C_CMP4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP4:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD4_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
  // HALO-NEXT:   %[[BD4_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP4]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL4:.*]] = stablehlo.select %[[BD4_PRED]], %[[CP4]], %[[BD4_VAL]] : tensor<1x4xi1>, tensor<1x4xf32>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %arg3, %[[SEL4]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<1x4xf32>, tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<5x4xf32>
  // HALO-NEXT:   %[[C_SSRC:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC:.*]] = stablehlo.multiply %[[REM]], %[[C_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN:.*]] = stablehlo.subtract %[[FIRST_WIN]], %[[ID_MUL_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW:.*]] = stablehlo.add %[[C_SRC_BASE]], %[[REL_FIRST_WIN]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[SRC_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<5x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // HALO-NEXT:   %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[SLICE1]], %[[SLICE2]], %arg4) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[COMP3:.*]] = stablehlo.compare GT, %arg6, %arg7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // HALO-NEXT:     stablehlo.return %[[COMP3]] : tensor<i1>
  // HALO-NEXT:   }, {
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
  // HALO-NEXT:     %[[ADD5:.*]] = stablehlo.add %arg6, %arg7 : tensor<f32>
  // HALO-NEXT:     stablehlo.return %[[ADD5]] : tensor<f32>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<6x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<6x4xf32>
  // HALO-NEXT:   %[[DS2:.*]] = stablehlo.dynamic_slice %[[SS]], %[[OUT_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<6x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  // HALO-NEXT:   sdy.return %[[DS2]] : tensor<4x4xf32>
  // HALO-NEXT: } : (tensor<8x8xf32>, tensor<6x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a":(2)2}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<6x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_passthrough
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<2x8xf32> {{.*}})
func.func @select_and_scatter_passthrough(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[RESHARD_SRC:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {"b"}]> : tensor<2x8xf32>
  // REPL-NEXT: %[[SS_REPL:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[RESHARD_SRC]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 4, 1>, window_strides = array<i64: 4, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<f32>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<f32>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xf32>, tensor<2x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // REPL-NEXT: %[[RES_REPL:.*]] = sdy.reshard %[[SS_REPL]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  // REPL-NEXT: return %[[RES_REPL]] : tensor<8x8xf32>

  // HALO-NOT:  sdy.manual_computation
  // HALO-NEXT: %[[SS_HALO:.*]] = "stablehlo.select_and_scatter"(%[[ARG0]], %[[ARG1]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 4, 1>, window_strides = array<i64: 4, 1>}> ({
  // HALO-NEXT: ^bb0(%[[H_ARG2:.*]]: tensor<f32>, %[[H_ARG3:.*]]: tensor<f32>):
  // HALO-NEXT:   %[[H_CMP:.*]] = stablehlo.compare GT, %[[H_ARG2]], %[[H_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // HALO-NEXT:   stablehlo.return %[[H_CMP]] : tensor<i1>
  // HALO-NEXT: }, {
  // HALO-NEXT: ^bb0(%[[H_ARG2:.*]]: tensor<f32>, %[[H_ARG3:.*]]: tensor<f32>):
  // HALO-NEXT:   %[[H_ADD:.*]] = stablehlo.add %[[H_ARG2]], %[[H_ARG3]] : tensor<f32>
  // HALO-NEXT:   stablehlo.return %[[H_ADD]] : tensor<f32>
  // HALO-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} : (tensor<8x8xf32>, tensor<2x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // HALO-NEXT: return %[[SS_HALO]] : tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 4, 1>,
    window_strides = array<i64: 4, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<2x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_negative_padding_sharded_fallback
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<5x8xf32>)
func.func @select_and_scatter_negative_padding_sharded_fallback(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<5x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // CHECK-NOT:  sdy.manual_computation
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // CHECK-SAME{LITERAL}: <{padding = dense<[[-1, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // CHECK-NEXT: ^bb0(%[[C_ARG2:.*]]: tensor<f32>, %[[C_ARG3:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[C_CMP:.*]] = stablehlo.compare GT, %[[C_ARG2]], %[[C_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK-NEXT:   stablehlo.return %[[C_CMP]] : tensor<i1>
  // CHECK-NEXT: }, {
  // CHECK-NEXT: ^bb0(%[[C_ARG2:.*]]: tensor<f32>, %[[C_ARG3:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[C_ADD:.*]] = stablehlo.add %[[C_ARG2]], %[[C_ARG3]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[C_ADD]] : tensor<f32>
  // CHECK-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xf32>, tensor<5x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[-1, 0], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<5x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_passthrough_unsharded_source_and_unused_pad_high
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<2x8xf32> {{.*}})
func.func @select_and_scatter_passthrough_unsharded_source_and_unused_pad_high(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[SS_REPL:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // REPL-SAME{LITERAL}: <{padding = dense<[[0, 2], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 4, 1>, window_strides = array<i64: 4, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f32>, %[[R_ARG3:.*]]: tensor<f32>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<f32>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<f32>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xf32>, tensor<2x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // REPL-NEXT: %[[RES_REPL:.*]] = sdy.reshard %[[SS_REPL]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  // REPL-NEXT: return %[[RES_REPL]] : tensor<8x8xf32>

  // HALO-NOT:  sdy.manual_computation
  // HALO-NEXT: %[[RESHARD_SRC:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<2x8xf32>
  // HALO-NEXT: %[[SS_HALO:.*]] = "stablehlo.select_and_scatter"(%[[ARG0]], %[[RESHARD_SRC]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 4, 1>, window_strides = array<i64: 4, 1>}> ({
  // HALO-NEXT: ^bb0(%[[H_ARG2:.*]]: tensor<f32>, %[[H_ARG3:.*]]: tensor<f32>):
  // HALO-NEXT:   %[[H_CMP:.*]] = stablehlo.compare GT, %[[H_ARG2]], %[[H_ARG3]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // HALO-NEXT:   stablehlo.return %[[H_CMP]] : tensor<i1>
  // HALO-NEXT: }, {
  // HALO-NEXT: ^bb0(%[[H_ARG2:.*]]: tensor<f32>, %[[H_ARG3:.*]]: tensor<f32>):
  // HALO-NEXT:   %[[H_ADD:.*]] = stablehlo.add %[[H_ARG2]], %[[H_ARG3]] : tensor<f32>
  // HALO-NEXT:   stablehlo.return %[[H_ADD]] : tensor<f32>
  // HALO-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} : (tensor<8x8xf32>, tensor<2x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // HALO-NEXT: return %[[SS_HALO]] : tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 4, 1>,
    window_strides = array<i64: 4, 1>,
    padding = dense<[[0, 2], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<2x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_unsigned_i32_gt
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32> {{.*}}, %[[ARG1:.*]]: tensor<6x8xi32>)
func.func @select_and_scatter_unsigned_i32_gt(
    %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<6x8xi32>)
    -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %cst = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[SS_REPL:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<i32>, %[[R_ARG3:.*]]: tensor<i32>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]], UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<i32>, %[[R_ARG3:.*]]: tensor<i32>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<i32>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xi32>, tensor<6x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SS_REPL]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xi32>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xi32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<i32>
  // HALO-NEXT: %[[UINT_MIN:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[UINT_MIN]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xi32>, %arg3: tensor<3x4xi32>, %arg4: tensor<i32>, %arg5: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN:.*]] = stablehlo.multiply %[[REM]], %[[C_SIN]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_WIN_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE:.*]] = stablehlo.multiply %[[FIRST_WIN]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_PAD_LOW]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW:.*]] = stablehlo.subtract %[[ID_PLUS_PAD]], %[[FIRST_WIN_MUL_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [2:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xi32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<2x4xi32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[C_CMP1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD1_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[BD1_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BD1_PRED]], %[[CP1]], %[[BD1_VAL]] : tensor<2x4xi1>, tensor<2x4xi32>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xi32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<2x4xi32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[C_CMP2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD2_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
  // HALO-NEXT:   %[[BD2_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BD2_PRED]], %[[CP2]], %[[BD2_VAL]] : tensor<2x4xi1>, tensor<2x4xi32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xi32>, tensor<4x4xi32>, tensor<2x4xi32>) -> tensor<8x4xi32>
  // HALO-NEXT:   %[[C_OP_BASE:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW:.*]] = stablehlo.subtract %[[C_OP_BASE]], %[[OUT_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[OP_OFFSET]], %c_1, sizes = [6, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xi32>, tensor<i32>, tensor<i32>) -> tensor<6x4xi32>
  // HALO-NEXT:   %[[SLICE_L_SRC:.*]] = stablehlo.slice %arg3 [2:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xi32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x4xi32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[C_CMP3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP3:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD3_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[BD3_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP3]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BD3_PRED]], %[[CP3]], %[[BD3_VAL]] : tensor<1x4xi1>, tensor<1x4xi32>
  // HALO-NEXT:   %[[SLICE_R_SRC:.*]] = stablehlo.slice %arg3 [0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xi32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x4xi32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[C_CMP4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP4:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD4_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[BD4_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP4]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL4:.*]] = stablehlo.select %[[BD4_PRED]], %[[CP4]], %[[BD4_VAL]] : tensor<1x4xi1>, tensor<1x4xi32>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %arg3, %[[SEL4]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<1x4xi32>, tensor<3x4xi32>, tensor<1x4xi32>) -> tensor<5x4xi32>
  // HALO-NEXT:   %[[C_SSRC:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC:.*]] = stablehlo.multiply %[[REM]], %[[C_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN:.*]] = stablehlo.subtract %[[FIRST_WIN]], %[[ID_MUL_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW:.*]] = stablehlo.add %[[C_SRC_BASE]], %[[REL_FIRST_WIN]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[SRC_OFFSET]], %c_1, sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<5x4xi32>, tensor<i32>, tensor<i32>) -> tensor<4x4xi32>
  // HALO-NEXT:   %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[SLICE1]], %[[SLICE2]], %arg4) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<i32>, %arg7: tensor<i32>):
  // HALO-NEXT:     %[[COMP3:.*]] = stablehlo.compare GT, %arg6, %arg7, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:     stablehlo.return %[[COMP3]] : tensor<i1>
  // HALO-NEXT:   }, {
  // HALO-NEXT:   ^bb0(%arg6: tensor<i32>, %arg7: tensor<i32>):
  // HALO-NEXT:     %[[ADD5:.*]] = stablehlo.add %arg6, %arg7 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD5]] : tensor<i32>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xi32>, tensor<4x4xi32>, tensor<i32>) -> tensor<6x4xi32>
  // HALO-NEXT:   %[[DS2:.*]] = stablehlo.dynamic_slice %[[SS]], %[[OUT_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xi32>, tensor<i32>, tensor<i32>) -> tensor<4x4xi32>
  // HALO-NEXT:   sdy.return %[[DS2]] : tensor<4x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<6x8xi32>, tensor<i32>, tensor<i32>) -> tensor<8x8xi32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare GT, %arg3, %arg4, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<6x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// CHECK-LABEL: func @select_and_scatter_f8E4M3FN_gt
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf8E4M3FN> {{.*}}, %[[ARG1:.*]]: tensor<6x8xf8E4M3FN>)
func.func @select_and_scatter_f8E4M3FN_gt(
    %arg0: tensor<8x8xf8E4M3FN> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<6x8xf8E4M3FN>)
    -> (tensor<8x8xf8E4M3FN> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f8E4M3FN>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f8E4M3FN>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf8E4M3FN>
  // REPL-NEXT: %[[SS_REPL:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f8E4M3FN>, %[[R_ARG3:.*]]: tensor<f8E4M3FN>):
  // REPL-NEXT:   %[[R_CMP:.*]] = stablehlo.compare GT, %[[R_ARG2]], %[[R_ARG3]] : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<i1>
  // REPL-NEXT:   stablehlo.return %[[R_CMP]] : tensor<i1>
  // REPL-NEXT: }, {
  // REPL-NEXT: ^bb0(%[[R_ARG2:.*]]: tensor<f8E4M3FN>, %[[R_ARG3:.*]]: tensor<f8E4M3FN>):
  // REPL-NEXT:   %[[R_ADD:.*]] = stablehlo.add %[[R_ARG2]], %[[R_ARG3]] : tensor<f8E4M3FN>
  // REPL-NEXT:   stablehlo.return %[[R_ADD]] : tensor<f8E4M3FN>
  // REPL-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xf8E4M3FN>, tensor<6x8xf8E4M3FN>, tensor<f8E4M3FN>) -> tensor<8x8xf8E4M3FN>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SS_REPL]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf8E4M3FN>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xf8E4M3FN>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<f8E4M3FN>
  // HALO-NEXT: %[[NEG_MAX:.*]] = stablehlo.constant dense<-4.480000e+02> : tensor<f8E4M3FN>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[NEG_MAX]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xf8E4M3FN>, %arg3: tensor<3x4xf8E4M3FN>, %arg4: tensor<f8E4M3FN>, %arg5: tensor<f8E4M3FN>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_SIN:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SIN:.*]] = stablehlo.multiply %[[REM]], %[[C_SIN]] : tensor<i32>
  // HALO-NEXT:   %[[C_WIN_OFFSET:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_NUM:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_WIN_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_CLAMP:.*]] = stablehlo.maximum %[[FIRST_WIN_NUM]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN:.*]] = stablehlo.divide %[[FIRST_WIN_CLAMP]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[FIRST_WIN_MUL_STRIDE:.*]] = stablehlo.multiply %[[FIRST_WIN]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[C_PAD_LOW:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[ID_PLUS_PAD:.*]] = stablehlo.add %[[ID_MUL_SIN]], %[[C_PAD_LOW]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET_RAW:.*]] = stablehlo.subtract %[[ID_PLUS_PAD]], %[[FIRST_WIN_MUL_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[OUT_OFFSET:.*]] = stablehlo.maximum %[[OUT_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg2 [2:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf8E4M3FN>) -> tensor<2x4xf8E4M3FN>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<2x4xf8E4M3FN>) -> tensor<2x4xf8E4M3FN>
  // HALO-NEXT:   %[[C_CMP1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD1_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f8E4M3FN>) -> tensor<2x4xf8E4M3FN>
  // HALO-NEXT:   %[[BD1_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BD1_PRED]], %[[CP1]], %[[BD1_VAL]] : tensor<2x4xi1>, tensor<2x4xf8E4M3FN>
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg2 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x4xf8E4M3FN>) -> tensor<2x4xf8E4M3FN>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<2x4xf8E4M3FN>) -> tensor<2x4xf8E4M3FN>
  // HALO-NEXT:   %[[C_CMP2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP2:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD2_VAL:.*]] = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f8E4M3FN>) -> tensor<2x4xf8E4M3FN>
  // HALO-NEXT:   %[[BD2_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [] : (tensor<i1>) -> tensor<2x4xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BD2_PRED]], %[[CP2]], %[[BD2_VAL]] : tensor<2x4xi1>, tensor<2x4xf8E4M3FN>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg2, %[[SEL2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xf8E4M3FN>, tensor<4x4xf8E4M3FN>, tensor<2x4xf8E4M3FN>) -> tensor<8x4xf8E4M3FN>
  // HALO-NEXT:   %[[C_OP_BASE:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET_RAW:.*]] = stablehlo.subtract %[[C_OP_BASE]], %[[OUT_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[OP_OFFSET:.*]] = stablehlo.maximum %[[OP_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[OP_OFFSET]], %[[C_ZERO1]], sizes = [6, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xf8E4M3FN>, tensor<i32>, tensor<i32>) -> tensor<6x4xf8E4M3FN>
  // HALO-NEXT:   %[[SLICE_L_SRC:.*]] = stablehlo.slice %arg3 [2:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xf8E4M3FN>) -> tensor<1x4xf8E4M3FN>
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> : (tensor<1x4xf8E4M3FN>) -> tensor<1x4xf8E4M3FN>
  // HALO-NEXT:   %[[C_CMP3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP3:.*]] = stablehlo.compare GE, %[[REM]], %[[C_CMP3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD3_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f8E4M3FN>) -> tensor<1x4xf8E4M3FN>
  // HALO-NEXT:   %[[BD3_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP3]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BD3_PRED]], %[[CP3]], %[[BD3_VAL]] : tensor<1x4xi1>, tensor<1x4xf8E4M3FN>
  // HALO-NEXT:   %[[SLICE_R_SRC:.*]] = stablehlo.slice %arg3 [0:1, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<3x4xf8E4M3FN>) -> tensor<1x4xf8E4M3FN>
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R_SRC]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}> : (tensor<1x4xf8E4M3FN>) -> tensor<1x4xf8E4M3FN>
  // HALO-NEXT:   %[[C_CMP4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP4:.*]] = stablehlo.compare LT, %[[REM]], %[[C_CMP4]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BD4_VAL:.*]] = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f8E4M3FN>) -> tensor<1x4xf8E4M3FN>
  // HALO-NEXT:   %[[BD4_PRED:.*]] = stablehlo.broadcast_in_dim %[[CMP4]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL4:.*]] = stablehlo.select %[[BD4_PRED]], %[[CP4]], %[[BD4_VAL]] : tensor<1x4xi1>, tensor<1x4xf8E4M3FN>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %arg3, %[[SEL4]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<1x4xf8E4M3FN>, tensor<3x4xf8E4M3FN>, tensor<1x4xf8E4M3FN>) -> tensor<5x4xf8E4M3FN>
  // HALO-NEXT:   %[[C_SSRC:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[ID_MUL_SSRC:.*]] = stablehlo.multiply %[[REM]], %[[C_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[REL_FIRST_WIN:.*]] = stablehlo.subtract %[[FIRST_WIN]], %[[ID_MUL_SSRC]] : tensor<i32>
  // HALO-NEXT:   %[[C_SRC_BASE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET_RAW:.*]] = stablehlo.add %[[C_SRC_BASE]], %[[REL_FIRST_WIN]] : tensor<i32>
  // HALO-NEXT:   %[[SRC_OFFSET:.*]] = stablehlo.maximum %[[SRC_OFFSET_RAW]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[SRC_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<5x4xf8E4M3FN>, tensor<i32>, tensor<i32>) -> tensor<4x4xf8E4M3FN>
  // HALO-NEXT:   %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[SLICE1]], %[[SLICE2]], %arg4) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<f8E4M3FN>, %arg7: tensor<f8E4M3FN>):
  // HALO-NEXT:     %[[COMP3:.*]] = stablehlo.compare GT, %arg6, %arg7 : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<i1>
  // HALO-NEXT:     stablehlo.return %[[COMP3]] : tensor<i1>
  // HALO-NEXT:   }, {
  // HALO-NEXT:   ^bb0(%arg6: tensor<f8E4M3FN>, %arg7: tensor<f8E4M3FN>):
  // HALO-NEXT:     %[[ADD5:.*]] = stablehlo.add %arg6, %arg7 : tensor<f8E4M3FN>
  // HALO-NEXT:     stablehlo.return %[[ADD5]] : tensor<f8E4M3FN>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xf8E4M3FN>, tensor<4x4xf8E4M3FN>, tensor<f8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // HALO-NEXT:   %[[DS2:.*]] = stablehlo.dynamic_slice %[[SS]], %[[OUT_OFFSET]], %[[C_ZERO1]], sizes = [4, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<6x4xf8E4M3FN>, tensor<i32>, tensor<i32>) -> tensor<4x4xf8E4M3FN>
  // HALO-NEXT:   sdy.return %[[DS2]] : tensor<4x4xf8E4M3FN>
  // HALO-NEXT: } : (tensor<8x8xf8E4M3FN>, tensor<6x8xf8E4M3FN>, tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<8x8xf8E4M3FN>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<f8E4M3FN>, %arg4: tensor<f8E4M3FN>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<f8E4M3FN>, %arg4: tensor<f8E4M3FN>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<f8E4M3FN>
      stablehlo.return %2 : tensor<f8E4M3FN>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf8E4M3FN>, tensor<6x8xf8E4M3FN>, tensor<f8E4M3FN>) -> tensor<8x8xf8E4M3FN>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf8E4M3FN>
  return %0 : tensor<8x8xf8E4M3FN>
}

// CHECK-LABEL: func @select_and_scatter_overlapping_windows_tight_right_halo
// CHECK-SAME: (%[[ARG0:.*]]: tensor<11x4xf32> {{.*}}, %[[ARG1:.*]]: tensor<6x2xf32>)
func.func @select_and_scatter_overlapping_windows_tight_right_halo(
    %arg0: tensor<11x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}]>},
    %arg1: tensor<6x2xf32>)
    -> (tensor<11x4xf32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {}]> : tensor<11x4xf32>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // REPL:      %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh_a4, [{"a"}, {}]> : tensor<11x4xf32>

  // HALO-NEXT: %[[RESHARD_SRC:.*]] = sdy.reshard %[[ARG1]] <@mesh_a4, [{"a"}, {}]> : tensor<6x2xf32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh_a4, []> : tensor<f32>
  // HALO-NEXT: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // HALO-NEXT: %[[PAD_IN:.*]] = stablehlo.pad %[[ARG0]], %[[NEG_INF]], low = [0, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}, {}]>]>} : (tensor<11x4xf32>, tensor<f32>) -> tensor<12x4xf32>
  // HALO-NEXT: %[[PAD_SRC:.*]] = stablehlo.pad %[[RESHARD_SRC]], %[[RESHARD_INIT]], low = [0, 0], high = [2, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}, {}]>]>} : (tensor<6x2xf32>, tensor<f32>) -> tensor<8x2xf32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[PAD_IN]], %[[PAD_SRC]], %[[RESHARD_INIT]], %[[NEG_INF]]) in_shardings=[<@mesh_a4, [{"a"}, {}]>, <@mesh_a4, [{"a"}, {}]>, <@mesh_a4, []>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a"}, {}]>] manual_axes={"a"} (%arg2: tensor<3x4xf32>, %arg3: tensor<2x2xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>) {
  // Operand uses leftHalo=2 and rightHalo=2 (concat 7 -> pad [1, 1] -> 9 -> slice 7).
  // HALO:      "stablehlo.collective_permute"({{.*}}) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]> : tensor<6x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO:      "stablehlo.collective_permute"({{.*}}) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5]]> : tensor<6x2xi64>}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO:      %[[OP_PAD:.*]] = stablehlo.pad {{.*}}, %arg5, low = [1, 0], high = [1, 0], interior = [0, 0] {{.*}} : (tensor<7x4xf32>, tensor<f32>) -> tensor<9x4xf32>
  // HALO:      %[[OP_SLICE:.*]] = stablehlo.dynamic_slice %[[OP_PAD]], {{.*}} sizes = [7, 4] {{.*}} : (tensor<9x4xf32>, tensor<i32>, tensor<i32>) -> tensor<7x4xf32>
  // Source only needs leftHalo=2 (rightHalo=0), padded high by 1 for safe dynamic_slice of size 3.
  // HALO:      %[[SRC_CP_L:.*]] = "stablehlo.collective_permute"({{.*}}) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]> : tensor<6x2xi64>}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // HALO-NOT:  stablehlo.collective_permute
  // HALO:      %[[SRC_CONCAT:.*]] = stablehlo.concatenate {{.*}} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
  // HALO:      %[[SRC_PAD:.*]] = stablehlo.pad %[[SRC_CONCAT]], %arg4, low = [0, 0], high = [1, 0], interior = [0, 0] {{.*}} : (tensor<4x2xf32>, tensor<f32>) -> tensor<5x2xf32>
  // HALO:      %[[SRC_SLICE:.*]] = stablehlo.dynamic_slice %[[SRC_PAD]], {{.*}} sizes = [3, 2] {{.*}} : (tensor<5x2xf32>, tensor<i32>, tensor<i32>) -> tensor<3x2xf32>
  // HALO:      "stablehlo.select_and_scatter"(%[[OP_SLICE]], %[[SRC_SLICE]], %arg4)
  // HALO:      sdy.return {{.*}} : tensor<3x4xf32>
  // HALO:      %[[RES:.*]] = stablehlo.slice %[[MC]] [0:11, 0:4] {{.*}} : (tensor<12x4xf32>) -> tensor<11x4xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %cmp = stablehlo.compare GT, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %sum = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %sum : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 3>,
    window_strides = array<i64: 2, 2>,
    padding = dense<1> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {}]>]>
  } : (tensor<11x4xf32>, tensor<6x2xf32>, tensor<f32>) -> tensor<11x4xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<11x4xf32>
  return %0 : tensor<11x4xf32>
}

// CHECK-LABEL: func @select_and_scatter_mixed_precision
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xbf16> {{.*}}, %[[ARG1:.*]]: tensor<6x8xf32>)
func.func @select_and_scatter_mixed_precision(
    %arg0: tensor<8x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<6x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xbf16>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // REPL:      %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xf32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<f32>
  // HALO-NEXT: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF80> : tensor<bf16>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[NEG_INF]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xbf16>, %arg3: tensor<3x4xf32>, %arg4: tensor<f32>, %arg5: tensor<bf16>) {
  // HALO:      "stablehlo.select_and_scatter"({{.*}}, {{.*}}, %arg4)
  // HALO:      : (tensor<6x4xbf16>, tensor<4x4xf32>, tensor<f32>) -> tensor<6x4xf32>
  // HALO:      sdy.return {{.*}} : tensor<4x4xf32>
  // HALO-NEXT: } : (tensor<8x8xbf16>, tensor<6x8xf32>, tensor<f32>, tensor<bf16>) -> tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
      %2 = stablehlo.compare GT, %arg2, %arg3 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xbf16>, tensor<6x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_missing_result_sharding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<6x8xf32>)
func.func @select_and_scatter_missing_result_sharding(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<6x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NOT:  sdy.manual_computation
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // CHECK:      return %[[SS]] : tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.compare GT, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>
  } : (tensor<8x8xf32>, tensor<6x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_dynamic_shape
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<?x8xf32>)
func.func @select_and_scatter_dynamic_shape(
    %arg0: tensor<?x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<?x8xf32>)
    -> (tensor<?x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NOT:  sdy.manual_computation
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<?x8xf32>
  // CHECK-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // CHECK:      %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<?x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.compare GT, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 2, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<?x8xf32>, tensor<?x8xf32>, tensor<f32>) -> tensor<?x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<?x8xf32>
  return %0 : tensor<?x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_synthetic_windows_nonzero_init_fallback
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<4x8xf32>)
func.func @select_and_scatter_synthetic_windows_nonzero_init_fallback(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<4x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NOT:  sdy.manual_computation
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // CHECK:      %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.compare GT, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 2, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<4x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_synthetic_windows_non_identity_scatter_fallback
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<4x8xf32>)
func.func @select_and_scatter_synthetic_windows_non_identity_scatter_fallback(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<4x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NOT:  sdy.manual_computation
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // CHECK:      %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.compare GT, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.subtract %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 2, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<4x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @select_and_scatter_synthetic_windows_mul_one_identity
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<4x8xf32>)
func.func @select_and_scatter_synthetic_windows_mul_one_identity(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<4x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = sdy.constant dense<1.000000e+00> : tensor<f32>
  %cst = sdy.constant dense<1.000000e+00> : tensor<f32>
  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_IN]], %[[ARG1]], %[[CST]])
  // REPL:      %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>

  // HALO-NEXT: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{"a"}, {"b"}]> : tensor<4x8xf32>
  // HALO-NEXT: %[[RESHARD_INIT:.*]] = sdy.reshard %[[CST]] <@mesh, []> : tensor<f32>
  // HALO-NEXT: %[[NEG_INF:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[RESHARD_OP]], %[[RESHARD_INIT]], %[[NEG_INF]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg2: tensor<4x4xf32>, %arg3: tensor<2x4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>) {
  // HALO:      "stablehlo.select_and_scatter"({{.*}}, {{.*}}, %arg4)
  // HALO:      stablehlo.multiply
  // HALO:      } : (tensor<8x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.compare GT, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.multiply %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 2, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xf32>, tensor<4x8xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

