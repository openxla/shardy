/* Copyright 2025 The MPMD Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "shardy/dialect/mpmd/transforms/optimize/pipeline_schedule.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

namespace mlir::mpmd {

std::optional<PipelineSchedule> ParsePipelineSchedule(StringRef schedule_str) {
  if (schedule_str.equals_insensitive("none")) {
    return PipelineSchedule::kNone;
  }
  if (schedule_str.equals_insensitive("1F1B")) {
    return PipelineSchedule::k1F1B;
  }
  if (schedule_str.equals_insensitive("GPipe")) {
    return PipelineSchedule::kGPipe;
  }
  if (schedule_str.equals_insensitive("Circular")) {
    return PipelineSchedule::kCircular;
  }
  if (schedule_str.equals_insensitive("CircularWithReversedBackward")) {
    return PipelineSchedule::kCircularWithReversedBackward;
  }
  if (schedule_str.equals_insensitive("GPipeBut1F1BForLastMesh")) {
    return PipelineSchedule::kGPipeBut1F1BForLastMesh;
  }
  if (schedule_str.equals_insensitive("ZeroBubbleH1")) {
    return PipelineSchedule::kZeroBubbleH1;
  }
  if (schedule_str.equals_insensitive("ZeroBubbleH2ZeroTxLatency")) {
    return PipelineSchedule::kZeroBubbleH2ZeroTxLatency;
  }
  if (schedule_str.equals_insensitive("ZeroBubbleH2HalfTxLatency")) {
    return PipelineSchedule::kZeroBubbleH2HalfTxLatency;
  }
  if (schedule_str.equals_insensitive("ZeroBubbleH2FullTxLatency")) {
    return PipelineSchedule::kZeroBubbleH2FullTxLatency;
  }
  if (schedule_str.equals_insensitive("ParallelPipelinesWithWrapAround")) {
    return PipelineSchedule::kParallelPipelinesWithWrapAround;
  }

  return std::nullopt;
}

std::string ToString(PipelineSchedule schedule) {
  switch (schedule) {
    case PipelineSchedule::kNone:
      return "none";
    case PipelineSchedule::k1F1B:
      return "1F1B";
    case PipelineSchedule::kGPipe:
      return "GPipe";
    case PipelineSchedule::kCircular:
      return "Circular";
    case PipelineSchedule::kCircularWithReversedBackward:
      return "CircularWithReversedBackward";
    case PipelineSchedule::kGPipeBut1F1BForLastMesh:
      return "GPipeBut1F1BForLastMesh";
    case PipelineSchedule::kZeroBubbleH1:
      return "ZeroBubbleH1";
    case PipelineSchedule::kZeroBubbleH2ZeroTxLatency:
      return "ZeroBubbleH2ZeroTxLatency";
    case PipelineSchedule::kZeroBubbleH2HalfTxLatency:
      return "ZeroBubbleH2HalfTxLatency";
    case PipelineSchedule::kZeroBubbleH2FullTxLatency:
      return "ZeroBubbleH2FullTxLatency";
    case PipelineSchedule::kParallelPipelinesWithWrapAround:
      return "ParallelPipelinesWithWrapAround";
  }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              FragmentComparatorOption comparator) {
  if (comparator.schedule) {
    return os << "builtin:" << ToString(*comparator.schedule);
  }
  return os << "custom";
}

namespace {

// Returns true if `fragment1` must happen before `fragment2` in a 1F1B
// schedule, or false otherwise.
// Requires: IsSchedulingUnit(fragment1) && IsSchedulingUnit(fragment2).
//
// Example of schedule obtained:
//  m0: F0 F1 F2 F3          B0 F4 B1 F5 B2    B3    B4    B5
//  m1:    F0 F1 F2       B0 F3 B1 F4 B2 F5 B3    B4    B5
//  m3:       F0 F1    B0 F2 B1 F3 B2 F4 B3 F5 B4    B5
//  m4:          F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5
// Where Fi stands for Forward stage of call_counter (i.e., microbatch) i and Bi
// stands for Backward of call_counter i.

// Based on 1F1B scheduling presented in PipeDream-2BW:
//   https://arxiv.org/abs/2006.09503
// Though note that this does not give any guarantees w.r.t. staleness.
bool OneFOneBMustHappenBefore(FragmentOp fragment1, FragmentOp fragment2) {
  // This guarantees that the transpose counts and call counts of each fragment
  // are defined.
  SDY_CHECK(IsSchedulingUnit(fragment1) && IsSchedulingUnit(fragment2));
  int64_t call_counter_f1 = *TryToFindCallCounter(fragment1);
  int64_t call_counter_f2 = *TryToFindCallCounter(fragment2);
  int64_t transpose_count_f1 = *TryToFindSingleTransposeCount(fragment1);
  int64_t transpose_count_f2 = *TryToFindSingleTransposeCount(fragment2);

  const int num_meshes = GetNumMeshes(fragment1);
  const int mesh_id = GetMeshIndex(fragment1);

  // The following two conditions guarantee the forward and backward fragments
  // are interleaved in the steady state of the pipeline.

  // Example: in mesh/stage 0 of pipeline of depth 4, the backward computation
  // of microbatch 0 must be scheduled before the forward computation of
  // microbatch 4: 0 == 4 - 4 + 0.
  if (transpose_count_f1 == 1 && transpose_count_f2 == 0) {
    return call_counter_f1 == call_counter_f2 - num_meshes + mesh_id;
  }

  // Example: in mesh/stage 0 of pipeline of depth 4, the forward computation of
  // microbatch 5 must be scheduled before the backward computation of
  // microbatch 2: 5 == 2 + 4 - (0 + 1).
  if (transpose_count_f1 == 0 && transpose_count_f2 == 1) {
    return call_counter_f1 == call_counter_f2 + num_meshes - (mesh_id + 1);
  }

  // If the fragments have the same transpose count, guarantee that the
  // call_counter ordering is preserved.
  if (transpose_count_f1 == transpose_count_f2) {
    return call_counter_f1 < call_counter_f2;
  }
  return false;
}

// Returns true if `fragment1` must happen before `fragment2` in a GPipe
// schedule, or false otherwise.
// Requires: IsSchedulingUnit(fragment1) && IsSchedulingUnit(fragment2).
// Based on https://arxiv.org/pdf/1811.06965.pdf.
bool GPipeMustHappenBefore(FragmentOp fragment1, FragmentOp fragment2) {
  // This guarantees that the transpose counts and call counts of each fragment
  // are defined.
  SDY_CHECK(IsSchedulingUnit(fragment1) && IsSchedulingUnit(fragment2));
  int call_counter_f1 = *TryToFindCallCounter(fragment1);
  int call_counter_f2 = *TryToFindCallCounter(fragment2);
  int transpose_count_f1 = *TryToFindSingleTransposeCount(fragment1);
  int transpose_count_f2 = *TryToFindSingleTransposeCount(fragment2);

  if (transpose_count_f1 == transpose_count_f2) {
    return call_counter_f1 < call_counter_f2;
  }
  return transpose_count_f1 < transpose_count_f2;
}

// Returns true if `fragment1` must happen before `fragment2` in a schedule that
// looks like GPipe in all meshes, except the last one, where we interleave Fwd
// and Bwd like in 1F1B. The advantage of this schedule is that we can avoid
// fragment remat in the last mesh.
// Example:
//   F0 F1 F2 F3       B0 B1    B2    B3
//      F0 F1 F2 F3 B0 B1    B2    B3
//         F0 B0 F1 B1 F2 B2 F3 B3
//
// Requires: IsSchedulingUnit(fragment1) && IsSchedulingUnit(fragment2).
bool GPipeBut1F1BLastMeshHappenBefore(FragmentOp fragment1,
                                      FragmentOp fragment2) {
  const int num_meshes = GetNumMeshes(fragment1);
  const int mesh_id = GetMeshIndex(fragment1);
  if (mesh_id == num_meshes - 1) {
    return OneFOneBMustHappenBefore(fragment1, fragment2);
  }
  return GPipeMustHappenBefore(fragment1, fragment2);
}

// This is the ZeroBubble H1 schedule from: https://arxiv.org/pdf/2401.10241.pdf
// F0  F1  F2  F3              Bᵃ0 Bʷ0 F4  Bᵃ1 Bʷ1 Bᵃ2 Bʷ2 Bᵃ3 Bʷ3 Bᵃ4 Bʷ4
//     F0  F1  F2          Bᵃ0 F3  Bᵃ1 Bʷ0 F4  Bᵃ2 Bʷ1 Bᵃ3 Bʷ2 Bᵃ4 Bʷ3 Bʷ4
//         F0  F1      Bᵃ0 F2  Bᵃ1 F3  Bᵃ2 Bʷ0 F4  Bᵃ3 Bʷ1 Bᵃ4 Bʷ2 Bʷ3 Bʷ4
//             F0  Bᵃ0 F1  Bᵃ1 F2  Bᵃ2 F3  Bᵃ3 Bʷ0 F4  Bᵃ4 Bʷ1 Bʷ2 Bʷ3 Bʷ4
// Namely the schedule relies on splitting the backwards pass into two
// computations: (i) backpropagation (Bᵃ above), and (ii) parameter gradient
// computation (Bʷ) above. The only small difference is that our splitting of
// backward fragments will not split the fragment on the last stage, since there
// are no transfers to other stages there.
bool ZeroBubbleH1MustHappenBefore(FragmentOp fragment1, FragmentOp fragment2) {
  // This guarantees that the transpose counts and call counts of each fragment
  // are defined.
  SDY_CHECK(IsSchedulingUnit(fragment1) && IsSchedulingUnit(fragment2));
  int64_t call_counter_f1 = *TryToFindCallCounter(fragment1);
  int64_t call_counter_f2 = *TryToFindCallCounter(fragment2);
  int64_t transpose_count_f1 = *TryToFindSingleTransposeCount(fragment1);
  int64_t transpose_count_f2 = *TryToFindSingleTransposeCount(fragment2);

  const int num_meshes = GetNumMeshes(fragment1);
  const int mesh_id = GetMeshIndex(fragment1);

  bool is_wgrad_f1 = IsSplitDropTransferred(fragment1);
  bool is_wgrad_f2 = IsSplitDropTransferred(fragment2);

  // The following two conditions guarantee the forward and backward fragments
  // are interleaved in the steady state of the pipeline. They are just like
  // 1F1B but specialized to actual back-propagation fragments.

  // Clause 1: Ba(i) < F(i + num_meshes - mesh_id)
  if (transpose_count_f1 == 1 && !is_wgrad_f1 && transpose_count_f2 == 0) {
    return call_counter_f1 == call_counter_f2 - num_meshes + mesh_id;
  }
  // Clause 2: F(i + num_meshes - mesh_id - 1) < Ba(i)
  if (transpose_count_f1 == 0 && transpose_count_f2 == 1 && !is_wgrad_f2) {
    return call_counter_f1 == call_counter_f2 + num_meshes - (mesh_id + 1);
  }

  // The rest of the conditions position the parameter gradient fragments.
  // Clause 3: Bw(i) < F(i + num_meshes)
  // e.g. Bw(0) < F(4) above.
  if (transpose_count_f1 == 1 && (is_wgrad_f1 || mesh_id == 0) &&
      transpose_count_f2 == 0) {
    return call_counter_f2 - call_counter_f1 == num_meshes;
  }
  // Clause 4: Ba(i + mesh_id) < Bw(i)
  // e.g.
  // mesh0:  Ba(0) < Bw(0)
  // mesh1:  Ba(1) < Bw(0)
  // mesh2:  Ba(2) < Bw(0)
  // mesh3:  Ba(3) < Bw(0)
  if (transpose_count_f1 == 1 && !is_wgrad_f1 && transpose_count_f2 == 1 &&
      is_wgrad_f2) {
    return call_counter_f1 - call_counter_f2 == mesh_id;
  }

  // This is just needed for transitively completing Clauses 3 and 2, needed for
  // the final phase where there may be no remaining forward to anchor to.
  // Bw(i) < Ba(i + mesh_id + 1)
  if (transpose_count_f1 == 1 && is_wgrad_f1 && transpose_count_f2 == 1 &&
      !is_wgrad_f2) {
    return call_counter_f2 - call_counter_f1 == mesh_id + 1;
  }

  return false;
}

// A function to calculate, for a given mesh, how many forward microbatches
// need to be streamed in, before we can schedule the first backward.
using InitFwdPerMeshFn = std::function<int(int)>;

bool ZeroBubbleH2MustHappenBefore(FragmentOp fragment1, FragmentOp fragment2,
                                  InitFwdPerMeshFn init_fwd_per_mesh) {
  SDY_CHECK(IsSchedulingUnit(fragment1) && IsSchedulingUnit(fragment2));
  int64_t call_counter_f1 = *TryToFindCallCounter(fragment1);
  int64_t call_counter_f2 = *TryToFindCallCounter(fragment2);
  int64_t transpose_count_f1 = *TryToFindSingleTransposeCount(fragment1);
  int64_t transpose_count_f2 = *TryToFindSingleTransposeCount(fragment2);

  const int num_meshes = GetNumMeshes(fragment1);
  const int mesh_id = GetMeshIndex(fragment1);

  bool is_wgrad_f1 = IsSplitDropTransferred(fragment1);
  bool is_wgrad_f2 = IsSplitDropTransferred(fragment2);

  // How many fwd we are allowed to stream before entering steady state.
  int init_fwd = init_fwd_per_mesh(mesh_id);
  // The ZeroBubbleH2 pipeline is diagonally symmetric (replacing forward with
  // backwards parameter gradient) so the following quantity is also part of the
  // schedule invariants below.
  int complement_init_fwd = init_fwd_per_mesh(num_meshes - mesh_id - 1);

  // Initial phase.
  // Clause 1: F(i) <= B(_) for i < init_fwd.
  if (transpose_count_f1 == 0 && transpose_count_f2 == 1 &&
      call_counter_f1 < init_fwd) {
    return true;
  }

  // Clause 2: Ba(i) < F(i + init_fwd)
  if (transpose_count_f1 == 1 && !is_wgrad_f1 && transpose_count_f2 == 0 &&
      call_counter_f2 >= init_fwd) {
    return call_counter_f2 - call_counter_f1 == init_fwd;
  }
  // Clause 3: F(i + init_fwd - 1) < Ba(i)
  if (transpose_count_f1 == 0 && call_counter_f1 >= init_fwd &&
      transpose_count_f2 == 1 && !is_wgrad_f2) {
    return call_counter_f1 - call_counter_f2 == init_fwd - 1;
  }

  // Clause 4: Ba(i + complement_init_fwd - 1) < Bw(i)
  if (transpose_count_f1 == 1 && !is_wgrad_f1 && transpose_count_f2 == 1 &&
      is_wgrad_f2) {
    return call_counter_f1 - call_counter_f2 == complement_init_fwd - 1;
  }
  // Clause 5: Bw(i) < Ba(i + complement_init_fwd)
  if (transpose_count_f1 == 1 && is_wgrad_f1 && transpose_count_f2 == 1 &&
      !is_wgrad_f2) {
    return call_counter_f2 - call_counter_f1 == complement_init_fwd;
  }
  return false;
}

// This is a variant of ZeroBubble H2 (https://arxiv.org/pdf/2401.10241.pdf.)
// The key difference is that the function below is equipped with a parameter
// `latency_stage_fraction` which specifies, as a float between 0.0f and 1.0f
// how much time activation forwarding transfers take compared to a stage
// compute time. For instance, a value 1.0f means that transfers take as much as
// a whole forward stage; a value 0.0f means that transfers are negligible.
//
// The way we use this number is to determine the 'eagerness' of the schedule
// per mesh, that is the number of forward micro-batches we need to stream in
// prior to the steady state in order to fully hide the pipeline bubble.
// We illustrate this with an example below:
//
// mesh0: fwd|tx                                  bwd
// mesh1:       fwd|tx                      bwd|tx
// mesh2:             fwd|tx          bwd|tx
// mesh3:                   fwd|bwd|tx
//        |<-----------------e0------------------>|
//              |<-----------e1----------->|
//                    |<-----e2------>|
//                          |e3|
//
// In particular e_i determines this number for each mesh index i. The function
// below analytically computes these numbers.
bool LatencyHidingZeroBubbleH2MustHappenBefore(float latency_stage_fraction,
                                               FragmentOp fragment1,
                                               FragmentOp fragment2) {
  SDY_CHECK_GE(latency_stage_fraction, 0.0f);
  SDY_CHECK_LE(latency_stage_fraction, 1.0f);

  int num_meshes = GetNumMeshes(fragment1);

  // The `init_fwds_per_mesh` returns the e_i in the diagram above, for
  // every mesh_i. This it the number of forward microbatches that can execute
  // before the first backwards microbatch can be executed on this mesh.
  auto init_fwds_per_mesh = [num_meshes, latency_stage_fraction](int mesh_id) {
    // The number of transfers from the beginning until the first backward
    // fragment can execute on mesh_id, see the diagram above. We call this the
    // "initial" path of the first microbatch in the pipeline.
    float num_init_transfers = 2.0f * (num_meshes - mesh_id - 1);
    // How much compute has happened in that initial first microbatch path, i.e.
    // until the point where the first backward fragment can execute on mesh_id.
    // The assumption that time(fwd) == time(bwd) (NB: this is just the backprop
    // bwd) may need to be revisited for real use cases.
    float num_init_compute = 2.0f * (num_meshes - mesh_id) - 1.0f;
    return std::floor(num_init_compute +
                      num_init_transfers * latency_stage_fraction);
  };
  return ZeroBubbleH2MustHappenBefore(fragment1, fragment2, init_fwds_per_mesh);
}

// Returns true if `fragment1` must happen before `fragment2` in a parallel
// pipeline with wrap-around situation.
//
// E.g. if we have 3 meshes and 3 microbatches. Note that the microbatch starts
// on different meshes.
//
// F1             F2    F3      <-- Mesh 1
//    F1    F2             F3   <-- Mesh 2
//       F1    F2    F3         <-- Mesh 3
// ~~>
// F1 F3 F2
// F2 F1 F3
// F3 F2 F1
//
// Requires:
// - IsSchedulingUnit(fragment1) && IsSchedulingUnit(fragment2)
// - Only forward fragments (in theory we could support backward fragments)
// - Mesh names are mesh1, mesh2, ..., mesh{n} and call_counter goes from 1 to
// n. We use this information for scheduling. Note that it's also ok to start
// from 0.
// - The entrypoint for mesh{i} is call_counter {i}
//
// Note that for each mesh, the order of fragments is the array
// [F{n}, F{n-1}, ..., F{1}] rotated such that
// the leading fragment is F{mesh_index}.
bool ParallelPipelinesWithWrapAroundMustHappenBefore(FragmentOp fragment1,
                                                     FragmentOp fragment2) {
  // This guarantees that the transpose counts and call counts of each fragment
  // are defined.
  SDY_CHECK(IsSchedulingUnit(fragment1) && IsSchedulingUnit(fragment2));
  // Only allowed for forwards for now.
  SDY_CHECK(IsForwardFragment(fragment1));
  SDY_CHECK(IsForwardFragment(fragment2));

  int64_t call_counter_f1 = *TryToFindCallCounter(fragment1);
  int64_t call_counter_f2 = *TryToFindCallCounter(fragment2);
  SDY_CHECK_NE(call_counter_f1, call_counter_f2)
      << "Should not have duplicate call counter.";

  int64_t mesh_num = 0;
  SDY_CHECK(llvm::to_integer(fragment1.getMeshName().drop_until(
                                 [](char c) { return llvm::isDigit(c); }),
                             mesh_num));
  // The entrypoint to mesh{i} is call_counter {i}, so this always happens
  // before.
  if (call_counter_f1 == mesh_num || call_counter_f2 == mesh_num) {
    return call_counter_f1 == mesh_num;
  }

  // `mesh_num` is the pivot. If both call_counters are on the same side of
  // the pivot, we flip the order. But if they are on different
  // sides, then we take the order as per normal.
  if ((call_counter_f1 > mesh_num && call_counter_f2 > mesh_num) ||
      (call_counter_f1 < mesh_num && call_counter_f2 < mesh_num)) {
    return call_counter_f1 > call_counter_f2;
  }
  return call_counter_f1 < call_counter_f2;
}

// Returns true if (f1, f2) are in (lexicographic) order, and false otherwise.
// If ascending is true then we compare the elements using < else >.
// Requires: f1 != f2.
bool LexicographicCompare(ArrayRef<int> f1, ArrayRef<int> f2, bool ascending) {
  auto in_order = [ascending](int a, int b) {
    return ascending ? a < b : a > b;
  };
  for (auto [f1_val, f2_val] : llvm::zip(f1, f2)) {
    if (in_order(f1_val, f2_val)) {
      return true;
    }
    if (f1_val == f2_val) continue;

    return false;
  }
  SDY_CHECK(false) << "Unreachable. We expect f1 != f2.";
}

bool CircularMustHappenBeforeBase(FragmentOp fragment1, FragmentOp fragment2,
                                  bool reversed_backward) {
  // This guarantees that the transpose counts and call counts of each fragment
  // are defined.
  SDY_CHECK(IsSchedulingUnit(fragment1) && IsSchedulingUnit(fragment2));
  const int call_counter_f1 = *TryToFindCallCounter(fragment1);
  const int call_counter_f2 = *TryToFindCallCounter(fragment2);
  const int transpose_count_f1 = *TryToFindSingleTransposeCount(fragment1);
  const int transpose_count_f2 = *TryToFindSingleTransposeCount(fragment2);

  if (!fragment1.getStageIdAttr() || !fragment2.getStageIdAttr()) {
    // Giving up. We cannot schedule for circular pipelining without stages.
    SDY_LOG(ERROR) << "Cannot schedule for circular pipelining without stages.";
    return false;
  }

  const int stage_f1 = fragment1.getStageIdAttr().getInt();
  const int stage_f2 = fragment2.getStageIdAttr().getInt();

  const int num_meshes = GetNumMeshes(fragment1);

  if (transpose_count_f1 != transpose_count_f2) {
    // Forward fragments always happen before backward fragments.
    return transpose_count_f1 < transpose_count_f2;
  }

  // transpose_count_f1 == transpose_count_f2, i.e., they're both forward *or*
  // both backward.

  // Given N meshes, with Circular pipelining, we want to execute N forward
  // fragments of the first logical stage assigned to a mesh, followed by N
  // forward fragments of the second logical stage assigned to the same mesh, as
  // so on, until all microbatches of all logical stages have been scheduled.
  // And then repeat this process for the backward fragments. We call a
  // subsequence of N fragments a *phase*.
  //
  // E.g., say you have a a model partitioned across 2 meshes (physical stages)
  // and 4 (logical) stages as follows (bwd pass omitted for simplicity):
  //
  // mesh0: A   C
  // mesh1:   B   D
  //
  // With 4 microbatches, we would get the following schedule:
  //
  // A1 A2 C1 C2 A3 A4 C3 C4
  //    B1 B2 D1 D2 B3 B4 D4 D4
  //
  // In a schedule of 2 meshes, we have phases of length 2.
  // The pipe `|` operator shows us the different phases of the schedule:
  // A1 A2 C1 C2 | A3 A4 C3 C4
  //    B1 B2 D1 D2 | B3 B4 D4 D4
  //
  // The phase of a fragment is defined as follows:
  const int phase_f1 = call_counter_f1 / num_meshes;
  const int phase_f2 = call_counter_f2 / num_meshes;

  SmallVector<int> f1 = {phase_f1, stage_f1, call_counter_f1};
  SmallVector<int> f2 = {phase_f2, stage_f2, call_counter_f2};

  // Fragments are first sorted by phase. If any two fragments are in the same
  // phase, then we sort them by stage. If they're in the same stage, then we
  // sort them by call_counter (or conceptually by call_counter % num_meshes).

  // Forward fragments.
  if (transpose_count_f1 == 0) {
    return LexicographicCompare(f1, f2, /*ascending=*/true);
  }

  // Backward fragments.
  if (reversed_backward) {
    return LexicographicCompare(f1, f2, /*ascending=*/false);
  }

  // Backward fragments with call_counter (and phase) in ascending order.
  // Naturally, the stage is in descending order (so we swap them before
  // comparison).
  std::swap(f1[1], f2[1]);
  return LexicographicCompare(f1, f2, /*ascending=*/true);
}

bool CircularWithReversedBackwardMustHappenBefore(FragmentOp fragment1,
                                                  FragmentOp fragment2) {
  return CircularMustHappenBeforeBase(fragment1, fragment2,
                                      /*reversed_backward=*/true);
}

bool CircularMustHappenBefore(FragmentOp fragment1, FragmentOp fragment2) {
  return CircularMustHappenBeforeBase(fragment1, fragment2,
                                      /*reversed_backward=*/false);
}

}  // namespace

FragmentComparator BuiltinFragmentComparator(PipelineSchedule schedule) {
  switch (schedule) {
    case PipelineSchedule::kNone: {
      return [](FragmentOp, FragmentOp) { return false; };
    }
    case PipelineSchedule::k1F1B: {
      return OneFOneBMustHappenBefore;
    }
    case PipelineSchedule::kGPipe: {
      return GPipeMustHappenBefore;
    }
    case PipelineSchedule::kGPipeBut1F1BForLastMesh: {
      return GPipeBut1F1BLastMeshHappenBefore;
    }
    case PipelineSchedule::kZeroBubbleH1: {
      return ZeroBubbleH1MustHappenBefore;
    }
    case PipelineSchedule::kZeroBubbleH2ZeroTxLatency: {
      return [](FragmentOp f1, FragmentOp f2) {
        return LatencyHidingZeroBubbleH2MustHappenBefore(0.0f, f1, f2);
      };
    }
    case PipelineSchedule::kZeroBubbleH2HalfTxLatency: {
      return [](FragmentOp f1, FragmentOp f2) {
        return LatencyHidingZeroBubbleH2MustHappenBefore(0.5f, f1, f2);
      };
    }
    case PipelineSchedule::kZeroBubbleH2FullTxLatency: {
      return [](FragmentOp f1, FragmentOp f2) {
        return LatencyHidingZeroBubbleH2MustHappenBefore(1.0f, f1, f2);
      };
    }
    case PipelineSchedule::kParallelPipelinesWithWrapAround: {
      return ParallelPipelinesWithWrapAroundMustHappenBefore;
    }
    case PipelineSchedule::kCircularWithReversedBackward: {
      return CircularWithReversedBackwardMustHappenBefore;
    }
    case PipelineSchedule::kCircular: {
      return CircularMustHappenBefore;
    }
  }
}

}  // namespace mlir::mpmd

namespace llvm::cl {

using ::mlir::mpmd::FragmentComparator;
using ::mlir::mpmd::FragmentComparatorOption;

template class basic_parser<FragmentComparatorOption>;

bool parser<FragmentComparatorOption>::parse(Option& opt, StringRef,
                                             StringRef arg,
                                             FragmentComparatorOption& value) {
  if (auto schedule = mlir::mpmd::ParsePipelineSchedule(arg)) {
    value.value = mlir::mpmd::BuiltinFragmentComparator(*schedule);
    value.schedule = schedule;
    return true;
  }
  return opt.error("unknown PropagationSchedule: " + arg);
}

void parser<FragmentComparatorOption>::printOptionDiff(
    const Option& opt, const FragmentComparatorOption& value,
    const OptVal& defaultValue, size_t globalWidth) const {
  printOptionName(opt, globalWidth);
  outs() << "= " << value << "\n";
}

void parser<FragmentComparatorOption>::anchor() {}

}  // namespace llvm::cl
