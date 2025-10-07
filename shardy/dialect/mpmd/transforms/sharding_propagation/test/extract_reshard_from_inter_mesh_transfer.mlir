// RUN: mpmd_opt %s -mpmd-extract-reshards-from-inter-mesh-transfers -split-input-file 2>&1 | FileCheck %s

module {
sdy.mesh @mesh = <["x"=2, "y"=2]>

// The local tensor at the destination has the same number of elements as the
// local tensor at the source.
// This causes the reshard to happen at consumer site.
// CHECK-LABEL: func @reshard_on_consumer_when_same_local_type_size
func.func @reshard_on_consumer_when_same_local_type_size(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>}
{
  // CHECK-NEXT: %[[TRANSFER_RESULT:.*]] = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  // CHECK-NEXT: %[[RESHARD_RESULT:.*]] = mpmd.fragment<mesh="m2", origin=[], out_shardings=[<@mesh, [{?}, {"x"}]>]> (%[[TRANSFER_RESULT]]) (%arg1: tensor<4x8xui32>) {
  // CHECK-NEXT:     mpmd.return %arg1 : tensor<4x8xui32>
  // CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x"}]>]>} %arg0: (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  func.return %0 : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
}
}

// -----

module {
sdy.mesh @mesh = <["x"=2, "y"=4]>

// The local tensor at the source has more elements than at the destination.
// This causes the reshard to happen at producer site.
// CHECK-LABEL: func @reshard_on_producer_when_local_type_smaller_on_producer
func.func @reshard_on_producer_when_local_type_smaller_on_producer(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  -> (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {?}]>}, !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["x"=2, "y"=4]>>>}
{
  // CHECK-NEXT: %[[RESHARD_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=[], out_shardings=[<@mesh, [{"y"}, {?}]>]> (%arg0) (%arg1: tensor<4x8xui32>) {
  // CHECK-NEXT:   mpmd.return %arg1 : tensor<4x8xui32>
  // CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  // CHECK-NEXT: %[[TRANSFER_RESULT:.*]] = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>} %0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  // CHECK-NEXT: return %[[TRANSFER_RESULT]], %arg0
  func.return %0, %arg0 : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
}
}

// -----

module {
sdy.mesh @mesh = <["x"=2, "y"=2]>

// Given the topology isn't homogeneous, no rewrite can be applied.
// CHECK-LABEL: func @topology_isnt_homogeneous
func.func @topology_isnt_homogeneous(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>, sharding=<@mesh, [{"x"}, {?}]>>)
  -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>}
{
  // CHECK-NEXT: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"y"}]>]>} %arg0
  // CHECK-NEXT: return
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"y"}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>, sharding=<@mesh, [{"x"}, {?}]>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  func.return %0 : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
}

// CHECK-LABEL: func @intra_mesh_transfer_introduces_reshard
func.func @intra_mesh_transfer_introduces_reshard(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>}
{
  // CHECK-NEXT: %[[TRANSFER_RESULT:.*]] = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  // CHECK-NEXT: %[[RESHARD_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=[], out_shardings=[<@mesh, [{?}, {"x"}]>]> (%[[TRANSFER_RESULT]]) (%arg1: tensor<4x8xui32>) {
  // CHECK-NEXT:    mpmd.return %arg1 : tensor<4x8xui32>
  // CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x"}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  func.return %0 : !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
}

// CHECK-LABEL: func @same_mesh_different_memory_kind
func.func @same_mesh_different_memory_kind(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>, memory_kind="pinned_host"> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>}
{
  // CHECK-NEXT: %[[TRANSFER_RESULT:.*]] = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %arg0 :
  // CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>, memory_kind="pinned_host">) ->
  // CHECK-SAME: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  // CHECK-NEXT: %[[RESHARD_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=[], out_shardings=[<@mesh, [{?}, {"x"}]>]> (%[[TRANSFER_RESULT]]) (%arg1: tensor<4x8xui32>)
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x"}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>, memory_kind="pinned_host">) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  func.return %0 : !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
}

// CHECK-LABEL: func @no_reshard_when_sharding_matches
func.func @no_reshard_when_sharding_matches(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>}
{
  // CHECK-NEXT: transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  func.return %0 : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
}

// CHECK-LABEL: func @no_reshard_on_null_and_replicated_sharding
func.func @no_reshard_on_null_and_replicated_sharding(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
  -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>}
{
  // CHECK-NOT: mpmd.fragment
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  func.return %0 : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
}

// All the tests above exercise the decision of whether to reshard at consumer
// or producer site, and no-op behaviour cases. The following tests focus on
// whether the value types should be immediately updated or inferred fragments
// should be created.
}

// -----
module {
sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @reshard_on_consumer_fragments_and_with_new_fragment
func.func @reshard_on_consumer_fragments_and_with_new_fragment(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  -> (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"x"}]>})
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>}
{
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>

  // Create a new reshard fragment as the transfer is used by another transfer
  // and by the return op.
  // CHECK-NEXT: %[[RESHARD:.*]] = mpmd.fragment<mesh="m2", origin=[], out_shardings=[<@mesh, [{?}, {"x"}]>]> (%[[TRANSFER]])
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %t = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x"}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>

  // Update the type of the fragment's operand with the resharded type.
  // CHECK-NEXT: %[[C1:.*]] = mpmd.fragment<mesh="m2", origin=[], in_shardings=[<@mesh, [{"x"}, {?}]>], out_shardings=[<@mesh, [{?}, {"x"}]>]>
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %c1 = mpmd.fragment<mesh="m2", origin=[], in_shardings=[<@mesh, [{?}, {"x"}]>], out_shardings=[<@mesh, [{?}, {"x"}]>]> (%t) (%arg1: tensor<4x8xui32>) {
    mpmd.return %arg1 : tensor<4x8xui32>
  } : (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>

  // CHECK-NEXT: %[[C2:.*]] = mpmd.fragment<mesh="m2", origin=[], in_shardings=[<@mesh, [{"x"}, {?}]>], out_shardings=[<@mesh, [{?}, {"x"}]>]>
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %c2 = mpmd.fragment<mesh="m2", origin=[], in_shardings=[<@mesh, [{?}, {"x"}]>], out_shardings=[<@mesh, [{?}, {"x"}]>]> (%t) (%arg1: tensor<4x8xui32>) {
    mpmd.return %arg1 : tensor<4x8xui32>
  } : (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>

  // Not a resharding transfer.
  // This transfer, if updated, would reshard the tensor, which isn't desired.
  // The resulting transfer must be identical to this one.
  // CHECK-NEXT: %[[C3:.*]] = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x"}]>]>} %[[RESHARD]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %c3 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x"}]>]>} %t : (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>

  // CHECK-NEXT: return %[[RESHARD]]
  func.return %t : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
}

}

// -----

module {
sdy.mesh @mesh = <["x"=2, "y"=4]>

// CHECK-LABEL: func @create_fragment_at_producer_when_used_by_another_transfer
func.func @create_fragment_at_producer_when_used_by_another_transfer(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"x"}]>})
  -> (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {?}]>}, !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["x"=2, "y"=4]>>>}
{
  // CHECK-NEXT: %[[PROD:.*]] = mpmd.fragment<mesh="m1", origin=[], in_shardings=[<@mesh, [{?}, {"x"}]>], out_shardings=[<@mesh, [{"y"}, {?}]>]> (%arg0)
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  %prod = mpmd.fragment<mesh="m1", origin=[], in_shardings=[<@mesh, [{?}, {"x"}]>], out_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<4x8xui32>) {
    mpmd.return %arg1 : tensor<4x8xui32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  // CHECK-NEXT: %[[RESHARD:.*]] =  mpmd.fragment<mesh="m1", origin=[], out_shardings=[<@mesh, [{"x"}, {?}]>]> (%[[PROD]])
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>} %[[PROD]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>} %prod : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>

  // Not a resharding transfer.
  // This transfer, if updated, would reshard the tensor, which isn't desired.
  // The resulting transfer must be identical to this one.
  // CHECK-NEXT: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %[[RESHARD]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  %another_user_of_prod = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %prod : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>

  // return %[[TRANSFER]], %[[RESHARD]]
  func.return %0, %prod : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
}

// CHECK-LABEL: func @create_fragment_when_producer_is_transfer
func.func @create_fragment_when_producer_is_transfer(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  -> (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {?}]>})
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["x"=2, "y"=4]>>>}
{
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  // CHECK-NEXT: %[[RESHARD:.*]] = mpmd.fragment<mesh="m1", origin=[], out_shardings=[<@mesh, [{"y"}, {?}]>]> (%[[TRANSFER]]) (%arg1: tensor<4x8xui32>) {
  // CHECK-NEXT:    mpmd.return %arg1 : tensor<4x8xui32>
  // CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  // CHECK-NEXT: %[[TRANSFER_RESULT:.*]] = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>} %1 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  %1 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>} %0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  func.return %1 : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
}

// CHECK-LABEL: func @update_input_sharding_on_consumer_fragment_with_existing_shardings
func.func @update_input_sharding_on_consumer_fragment_with_existing_shardings(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  -> (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"x"}]>})
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>}
{
  // CHECK-NEXT: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x"}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  // CHECK-NEXT: mpmd.fragment<mesh="m2", origin=[], in_shardings=[<@mesh, [{"x"}, {?}]>], out_shardings=[<@mesh, [{?}, {"x"}]>]>
  %consumer = mpmd.fragment<mesh="m2", origin=[], in_shardings=[<@mesh, [{?}, {"x"}]>], out_shardings=[<@mesh, [{?}, {"x"}]>]> (%0) (%arg1: tensor<4x8xui32>) {
    mpmd.return %arg1 : tensor<4x8xui32>
  } : (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  func.return %consumer : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
}

// CHECK-LABEL: func @update_input_sharding_on_consumer_fragment_with_no_existing_shardings
func.func @update_input_sharding_on_consumer_fragment_with_no_existing_shardings(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  -> (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"x"}]>})
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>}
{
  // CHECK-NEXT: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x"}]>]>} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  // CHECK-NEXT: mpmd.fragment<mesh="m2", origin=[], in_shardings=[<@mesh, [{"x"}, {?}]>], out_shardings=[<@mesh, [{?}, {"x"}]>]>
  %consumer = mpmd.fragment<mesh="m2", origin=[], out_shardings=[<@mesh, [{?}, {"x"}]>]> (%0) (%arg1: tensor<4x8xui32>) {
    mpmd.return %arg1 : tensor<4x8xui32>
  } : (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  func.return %consumer : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
}

// CHECK-LABEL: func @update_sharding_on_producer_fragment
func.func @update_sharding_on_producer_fragment(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>})
  -> (!mpmd.mesh_tensor<"m2", tensor<4x8xui32>> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {?}]>})
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["x"=2, "y"=4]>>>}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["prod"], in_shardings=[<@mesh, [{?}, {"x"}]>], out_shardings=[<@mesh, [{"y"}, {?}]>]> (%arg0) (%arg1: tensor<4x8xui32>) {
  // CHECK-NEXT:    mpmd.return %arg1 : tensor<4x8xui32>
  // CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  // CHECK-NEXT: mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>} %0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  %producer = mpmd.fragment<mesh="m1", origin=["prod"], in_shardings=[<@mesh, [{?}, {"x"}]>], out_shardings=[<@mesh, [{?}, {"x"}]>]> (%arg0) (%arg1: tensor<4x8xui32>) {
    mpmd.return %arg1 : tensor<4x8xui32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xui32>>
  %0 = mpmd.transfer {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>} %producer : (!mpmd.mesh_tensor<"m1", tensor<4x8xui32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
  func.return %0 : !mpmd.mesh_tensor<"m2", tensor<4x8xui32>>
}

}
