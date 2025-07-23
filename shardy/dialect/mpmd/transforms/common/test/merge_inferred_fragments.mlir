// RUN: mpmd_opt %s -mpmd-merge-inferred-fragments 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @producer_user_is_inter_mesh_transfer_consumer_is_inferred
func.func @producer_user_is_inter_mesh_transfer_consumer_is_inferred(
  %arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT: %[[MERGED_FRAGMENT:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
// CHECK:      %[[TRANSFER:.*]] = mpmd.transfer %[[MERGED_FRAGMENT]]#0
// CHECK-NEXT: mpmd.fragment<mesh="m2", origin=["use_transfer"]> (%[[TRANSFER]])
  %0:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4, %arg3 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0#0)
    (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %2 = mpmd.transfer %0#1 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  // If the transfer was only used by the func return it wouldn't block merging.
  %3 = mpmd.fragment<mesh="m2", origin=["use_transfer"]> (%2)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_2_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  func.return %1, %3 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @noop_fragment_doesnt_prevent_merging
func.func @noop_fragment_doesnt_prevent_merging(
  %arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // Note: when merging two fragments, in which one of them is a noop, we use
  // the name of the other fragment as a name of the resulting fragment.
  // CHECK-NEXT: %[[MERGED_FRAGMENT:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
  // CHECK:      %[[TRANSFER:.*]] = mpmd.transfer %[[MERGED_FRAGMENT]]#0
  // CHECK-NEXT: mpmd.fragment<mesh="m2", origin=["use_transfer"]> (%[[TRANSFER]])
  %0:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4, %arg3 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0#0)
    (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %2 = mpmd.transfer %0#1 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  // If the transfer was only used by the func return it wouldn't block merging.
  %3 = mpmd.fragment<mesh="m2", origin=["use_transfer"]> (%2)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_2_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  func.return %1, %3 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @consumer_operand_is_inter_mesh_transfer
func.func @consumer_operand_is_inter_mesh_transfer_producer_is_inferred(
  %arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg1
// CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %[[TRANSFER]])
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.transfer %arg1 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %2 = mpmd.fragment<mesh="m1", origin=["f"]> (%0, %1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @both_fragments_are_negligible
func.func @both_fragments_are_negligible(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT: %[[MERGED_FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   mpmd.return %arg1
// CHECK-NEXT: }
// CHECK-NEXT: return %[[MERGED_FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @consumer_origin_is_empty
func.func @consumer_origin_is_empty(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT: %[[MERGED_FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   mpmd.return %arg1
// CHECK-NEXT: }
// CHECK-NEXT: return %[[MERGED_FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @origins_are_empty
func.func @origins_are_empty(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT: %[[MERGED_FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   mpmd.return %arg1
// CHECK-NEXT: }
// CHECK-NEXT: return %[[MERGED_FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// Same as @small_inferred_fragment_is_cloned in mpmd_merge_fragments_with_cloning.mlir.
// However, the pass tested in this file does not allow for fragments to be cloned.

// CHECK-LABEL: func @small_inferred_fragment_is_not_cloned
func.func @small_inferred_fragment_is_not_cloned(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: %[[FRAG1:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply
  // CHECK-NEXT:   return %[[ADD]], %[[MULT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[FRAG2:.*]] = mpmd.fragment<mesh="m1", origin=["g"]> (%[[FRAG1]]#0)
  // CHECK-NEXT:   stablehlo.add
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[FRAG1]]#1, %[[FRAG2]]
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.fragment<mesh="m1", origin=["g"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %1, %2 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @inferred_producer_can_be_merged_into_stage
func.func @inferred_producer_can_be_merged_into_stage(%arg0: !mesh_1_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%arg0)
  // CHECK-NOT: mpmd.fragment
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @stage_can_be_merged_into_inferred_consumer
func.func @stage_can_be_merged_into_inferred_consumer(%arg0: !mesh_1_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%arg0)
  // CHECK-NOT: mpmd.fragment
  %0 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @should_merge_if_producer_result_used_by_another_mesh
func.func @should_merge_if_producer_result_used_by_another_mesh(
  %arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: %[[MERGED_PRODUCER_RESULT:.*]]:2 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT:   %[[ADD2:.*]] = stablehlo.add
  // CHECK-NEXT:   return %[[ADD]], %[[ADD2:.*]]
  // CHECK-NEXT: }
  %produced_result = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  // CHECK-NEXT:      %[[TRANSFER:.*]] = mpmd.transfer %[[MERGED_PRODUCER_RESULT]]#
  %result_transferred_to_another_mesh = mpmd.transfer %produced_result : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  // CHECK-NEXT:      mpmd.fragment<mesh="m2", origin=["diff_mesh_consumer"]>
  %result_from_diff_mesh = mpmd.fragment<mesh="m2", origin=["diff_mesh_consumer"]> (%result_transferred_to_another_mesh)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_2_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  %result_from_inferred_consumer = mpmd.fragment<mesh="m1", origin=[]> (%produced_result)
    (%arg2: tensor<4x8xf32>) {
    %5 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %5 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %result_from_inferred_consumer, %result_from_diff_mesh : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @can_merge_if_all_consumer_operands_produced_before_producer
func.func @can_merge_if_all_consumer_operands_produced_before_producer(
  %arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: %[[EARLIER_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["older_than_producer"]> (%arg0)
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT:   return %[[ADD]]
  // CHECK-NEXT: }
  // We don't merge the %earlier_result fragment with the inferred consumer because it would reorder the fragments.
  %earlier_result = mpmd.fragment<mesh="m1", origin=["older_than_producer"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK-NEXT: %[[MERGED_PRODUCER_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]> (%[[EARLIER_RESULT]], %arg0)
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT:   %[[ADD2:.*]] = stablehlo.add
  // CHECK-NEXT:   return %[[ADD2:.*]]
  // CHECK-NEXT: }
  %later_result = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %final_result = mpmd.fragment<mesh="m1", origin=[]> (%earlier_result, %later_result) // inferred consumer.
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %5 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %5 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %final_result : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @should_merge_into_producer_if_producer_results_used_before_consumer
func.func @should_merge_into_producer_if_producer_results_used_before_consumer(
  %arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: %[[MERGED_FRAGMENT:.*]]:2 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT:   %[[ADD2:.*]] = stablehlo.add
  // CHECK-NEXT:   return %[[ADD:.*]], %[[ADD2:.*]]
  // CHECK-NEXT: }
  %producer_result = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %[[MERGED_FRAGMENT]]#
  %transfer_result = mpmd.transfer %producer_result : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  %final_result = mpmd.fragment<mesh="m1", origin=[]> (%producer_result) // inferred consumer.
    (%arg2: tensor<4x8xf32>) {
    %5 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %5 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %final_result : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @should_merge_into_consumer_if_consumer_uses_result_in_between
func.func @should_merge_into_consumer_if_consumer_uses_result_in_between(
  %arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg0
  // CHECK-NEXT: %[[TRANSFER2:.*]] = mpmd.transfer %[[TRANSFER]]
  %producer_result = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %transfer_result_1 = mpmd.transfer %arg0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  %transfer_result_2 = mpmd.transfer %transfer_result_1 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK-NEXT: %[[MERGED_FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0, %[[TRANSFER2]])
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT:   %[[ADD2:.*]] = stablehlo.add
  // CHECK-NEXT:   return %[[ADD2]]
  // CHECK-NEXT: }
  %final_result = mpmd.fragment<mesh="m1", origin=[]> (%producer_result, %transfer_result_2) // inferred consumer.
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %5 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %5 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %final_result : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @should_not_merge_if_it_would_create_cyclic_dependency
func.func @should_not_merge_if_it_would_create_cyclic_dependency(
  %arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: %[[PRODUCER_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT:   return %[[ADD]]
  // CHECK-NEXT: }
  %producer_result = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK-NEXT: %[[TRANSFER_RESULT:.*]] = mpmd.transfer %[[PRODUCER_RESULT]]
  // CHECK-NEXT: %[[TRANSFER2_RESULT:.*]] = mpmd.transfer %[[TRANSFER_RESULT]]
  %transfer_result_1 = mpmd.transfer %producer_result : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  %transfer_result_2 = mpmd.transfer %transfer_result_1 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=[]> (%[[PRODUCER_RESULT]], %[[TRANSFER2_RESULT]])
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add
  // CHECK-NEXT:   return %[[ADD]]
  // CHECK-NEXT: }
  %final_result = mpmd.fragment<mesh="m1", origin=[]> (%producer_result, %transfer_result_2) // inferred consumer.
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %5 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %5 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %final_result : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @call_counter_user_is_preferred_for_consumer_user
func.func @call_counter_user_is_preferred_for_consumer_user(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT:  {call_counter = 1 : ui32}
// CHECK-NOT: call_counter
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) {call_counter = 0 : ui32}
    (%arg1: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%0) {call_counter = 1 : ui32}
    (%arg1: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @call_counter_user_is_preferred_for_producer_user
func.func @call_counter_user_is_preferred_for_producer_user(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT: {call_counter = 0 : ui32}
// CHECK-NOT: call_counter
  %0 = mpmd.fragment<mesh="m1", origin=["f0"]> (%arg0) {call_counter = 0 : ui32}
    (%arg1: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0) {call_counter = 1 : ui32}
    (%arg1: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @call_counter_removed_if_both_inferred
func.func @call_counter_removed_if_both_inferred(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NOT: call_counter
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) {call_counter = 0 : ui32}
    (%arg1: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0) {call_counter = 1 : ui32}
    (%arg1: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}
