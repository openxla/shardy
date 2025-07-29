// RUN: mpmd_opt %s -mpmd-absorb-inferred-fragments 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// The tests file exercise absortion of inferred fragments by User Defined
// Fragments (UDFs) and other root fragments.

// CHECK-LABEL: func private @simple_udf_absorbs_inferred_consumer_and_preserves_call_counter
func.func private @simple_udf_absorbs_inferred_consumer_and_preserves_call_counter
 (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1) {call_counter = 123 : ui32}
  // CHECK-NEXT:   add
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   return
  // CHECK-NOT:  origin=[]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1) {call_counter = 123 : ui32} (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // Note: this transfer is the first consumer of the UDF. This won't prevent
  // merging from happening though.
  %transfer = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1, %transfer : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func private @single_udf_producer_merges_all_inferred_consumers
func.func private @single_udf_producer_merges_all_inferred_consumers
  (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   add
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   return
  // CHECK-NOT:  origin=[]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %2 = mpmd.fragment<mesh="m1", origin=[]> (%0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %3 = mpmd.fragment<mesh="m1", origin=[]> (%0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1, %2, %3 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func private @udf_absorbs_consumer_after_transfer_is_moved_to_block_begin
func.func private @udf_absorbs_consumer_after_transfer_is_moved_to_block_begin
  (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: transfer %arg1
  // CHECK-NEXT: fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   add
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: transfer
  // CHECK-NOT:  origin=[]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // Note: this transfer is the first consumer of the UDF. This won't prevent
  // merging from happening though.
  %transfer = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  // This is actually the closest producer of the inferred fragment. However,
  // it doesn't depend on the UDF and it can be moved before the UDF.
  %arg1_transfer = mpmd.transfer %arg1 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0, %arg1_transfer) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1, %transfer : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// Same as above, but closest producer is a transfer of an inferred fragment,
// not a block argument.
// CHECK-LABEL: func private @udf_absorbs_consumer_after_transfer_is_moved_to_producer
func.func private @udf_absorbs_consumer_after_transfer_is_moved_to_producer
  (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: %[[C:.*]] = mpmd.fragment<mesh="m2", origin=[]> () ()
  // CHECK-NEXT:   constant
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: transfer %[[C]]
  // CHECK-NEXT: fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   add
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: transfer
  // CHECK-NOT:  origin=[]

  // There's nothing to absorb this inferred fragment, so it will remain in the
  // final program.
  %c = mpmd.fragment<mesh="m2", origin=[]> () () {
    %4 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : () -> !mesh_2_tensor_4_8_f32

  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // Note: this transfer is the first consumer of the UDF. This won't prevent
  // merging from happening though.
  %transfer = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  // This is actually the closest producer of the inferred fragment. However,
  // it doesn't depend on the UDF and it can be moved before the UDF.
  %c_transfer = mpmd.transfer %c : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0, %c_transfer) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1, %transfer : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// Same as above, but %c is defined after the udf. In this test nothing gets
// absorbed (see explanation below).
// CHECK-LABEL: func private @udf_cannot_absorb_because_transfer_cannot_be_moved
func.func private @udf_cannot_absorb_because_transfer_cannot_be_moved(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
  // CHECK-NEXT:   add
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // CHECK-NEXT: %[[C:.*]] = mpmd.fragment<mesh="m2", origin=[]> () ()
  // CHECK-NEXT:   constant
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  %c = mpmd.fragment<mesh="m2", origin=[]> () () {
    %4 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : () -> !mesh_2_tensor_4_8_f32

  // This is actually the closest producer of the inferred fragment. Although it
  // doesn't depend on the UDF, it cannot be moved before the UDF because it is
  // defined after the UDF. And even though there's no reason for %c to be
  // defined where it is, we haven't yet implemented logic to move it around.
  // TODO: b/370062636 - We should change the merging algorithm to check if
  // there are any dependencies between values, instead of simply relying on
  // program order.
  // CHECK-NEXT: transfer %[[C]]
  %c_transfer = mpmd.transfer %c : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // CHECK-NEXT: fragment<mesh="m1", origin=[]>
  // CHECK-NEXT:   multiply
  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0, %c_transfer) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func private @inferred_root_absorbs_inferred_producer_and_then_is_absorbed_by_udf
func.func private @inferred_root_absorbs_inferred_producer_and_then_is_absorbed_by_udf
  (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
  // CHECK-NEXT:   constant
  // CHECK-NEXT:   add
  // CHECK-NEXT:   divide
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   return
  // CHECK-NOT:  origin=[]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // This is the closest producer of the inferred consumer of the UDF. However,
  // the inferred consumer is also a root fragment (only used by the consumer),
  // so everything will be merged.
  %1 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.constant dense<2.0> : tensor<4x8xf32>
    %5 = stablehlo.divide %arg2, %4 : tensor<4x8xf32>
    mpmd.return %5 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %2 = mpmd.fragment<mesh="m1", origin=[]> (%0, %1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func private @closest_of_two_udf_producers_absorbs_inferred_consumer
func.func private @closest_of_two_udf_producers_absorbs_inferred_consumer
  (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: %[[F:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
  // CHECK-NEXT:   add
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: fragment<mesh="m1", origin=["g"]> (%arg0, %arg1, %[[F]])
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   divide
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NOT: origin=[]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["g"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // Although it has two UDF fragments as producers, it must be merged to the
  // one that it's closer to, i.e., g.
  %2 = mpmd.fragment<mesh="m1", origin=[]> (%0, %1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.divide %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func private @simple_udf_absorbs_inferred_producer
func.func private @simple_udf_absorbs_inferred_producer
  (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: transfer
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   add
  // CHECK-NEXT:   return
  // CHECK-NOT:  origin=[]
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // Note this transfer is the first producer of the UDF. This won't prevent
  // merging from happening though.
  %arg1_transfer = mpmd.transfer %arg1 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg1_transfer, %0) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func private @udf_absorbs_multiple_inferred_producers
func.func private @udf_absorbs_multiple_inferred_producers
  (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: transfer
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   add
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   divide
  // CHECK-NEXT:   return
  // CHECK-NOT:  origin=[]
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %arg1_transfer = mpmd.transfer %arg1 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg1_transfer, %0, %1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %4 = stablehlo.divide %arg3, %arg4 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func private @closest_of_two_udf_consumers_absorb_inferred_producer
func.func private @closest_of_two_udf_consumers_absorb_inferred_producer
  (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   add
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["g"]>
  // CHECK-NEXT:   divide
  // CHECK-NEXT:   return
  // CHECK-NOT:  origin=[]
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"]> (%0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %2 = mpmd.fragment<mesh="m1", origin=["g"]> (%0, %1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.divide %arg3, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %2 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func private @delay_transfer_to_return_so_udf_absorbs_inferred_producer
func.func private @delay_transfer_to_return_so_udf_absorbs_inferred_producer
  (%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   divide
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: transfer
  // CHECK-NOT:  origin=[]
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // This is the closest consumer of the inferred fragment. However, it's not
  // needed by the udf and can be postponed.
  %transfer = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  %2 = mpmd.fragment<mesh="m1", origin=["f"]> (%0, %arg0) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.divide %arg3, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %2, %transfer : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func private @udf_cannot_absorb_because_transfer_is_closest_consumer_and_cannot_be_postponed
func.func private @udf_cannot_absorb_because_transfer_is_closest_consumer_and_cannot_be_postponed
  (%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=[]>
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: transfer
  // CHECK-NEXT: transfer
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   divide
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // This is the closest consumer of the inferred fragment and the closest
  // consumer cannot be postponed. The program stays as it is.
  %transfer_to_m2 = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  %transfer_to_m1 = mpmd.transfer %transfer_to_m2 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %2 = mpmd.fragment<mesh="m1", origin=["f"]> (%0, %arg0) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.divide %arg3, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %2, %transfer_to_m1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func private @transfer_gets_removed_so_udf_absorbs_inferred_producer
func.func private @transfer_gets_removed_so_udf_absorbs_inferred_producer
  (%arg0: !mesh_1_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   divide
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NOT: transfer
  // CHECK-NOT:  origin=[]
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // This is the closest consumer of the inferred fragment. However, it's not
  // needed by the udf and can be postponed.
  %transfer = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  %2 = mpmd.fragment<mesh="m1", origin=["f"]> (%0, %arg0) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.divide %arg3, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %2 : !mesh_1_tensor_4_8_f32
}
