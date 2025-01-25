/* Copyright 2024 The Shardy Authors.

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

#include <cassert>
#include <cstdint>
#include <iterator>
#include <list>
#include <memory>  // IWYU pragma: keep
#include <optional>
#include <set>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_RESHARDTOCOLLECTIVESPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

using OptionalAxisRef = std::optional<AxisRefAttr>;

// We use an std::list so we can pop from the front and the back and with an
// iterator at constant time.
// TODO(tomnatan): Consider using AxisListRef instead of std::list once it can
// also replace the first axis in the list with a different sub-axis.
using AxisList = std::list<AxisRefAttr>;

// We use an std::set so sub-axes are ordered by their pre-size and size, and
// we can use set::lower_bound to find the first overlapping axis (see
// getFirstOverlapping).
using AvailableAxes = std::set<AxisRefAttr>;

// Removes the common prefix of both `first` and `second`.
void removeCommonPrefix(AxisList& first, AxisList& second, MeshAttr mesh) {
  while (!first.empty() && !second.empty() && first.front() == second.front()) {
    first.pop_front();
    second.pop_front();
  }
  if (first.empty() || second.empty()) {
    return;
  }
  if (OptionalAxisRef suffix =
          first.front().removeCommonPrefix(second.front(), mesh)) {
    first.front() = *suffix;
    second.pop_front();
  } else if (OptionalAxisRef suffix =
                 second.front().removeCommonPrefix(first.front(), mesh)) {
    second.front() = *suffix;
    first.pop_front();
  }
}

// Returns a vector of `InnerAxisList` per dimension from the given `sharding`.
template <class InnerAxisList>
SmallVector<InnerAxisList> getAxesPerDim(TensorShardingAttr sharding) {
  SmallVector<InnerAxisList> axesPerDim;
  axesPerDim.reserve(sharding.getRank());
  for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
    axesPerDim.emplace_back(dimSharding.axis_begin(), dimSharding.axis_end());
  }
  return axesPerDim;
}

AvailableAxes::const_iterator getPrevOrEnd(AvailableAxes::iterator it,
                                           const AvailableAxes& availableAxes) {
  return it == availableAxes.begin() ? availableAxes.end() : std::prev(it);
}

// Returns an iterator to the first axis in `availableAxes` that overlaps with
// `axis`, or `availableAxes.end()` if there is no such axis.
AvailableAxes::iterator getFirstOverlapping(
    AxisRefAttr axis, const AvailableAxes& availableAxes) {
  if (availableAxes.empty()) {
    return availableAxes.end();
  }
  auto afterIt = availableAxes.lower_bound(axis);
  auto beforeIt = getPrevOrEnd(afterIt, availableAxes);
  // If there is at least one overlapping axis, the first one is necessarily
  // `afterIt` or `beforeIt`.
  //
  // Proof:
  // Let `axis` be A and the first overlapping axis in `availableAxes` be B.
  //
  // Note that there can't be two overlapping available axes. `lower_bound`
  // returns the first available axis greater or equal to A.
  //
  // * If `B >= A`, then there can't be another available axis C such that
  //   `A <= C < B` since it would have to be overlapping with A and thus the
  //   first overlapping axis instead of B. Therefore, `lower_bound` will
  //   return B.
  // * If `B < A`, then there can't be another available axis C such that
  //   `B < C < A` since B and C can't overlap. Therefore, `lower_bound` will
  //   return the axis after B, which doesn't overlap with A.

  if (beforeIt != availableAxes.end() && beforeIt->overlaps(axis)) {
    return beforeIt;
  }
  if (afterIt != availableAxes.end() && afterIt->overlaps(axis)) {
    return afterIt;
  }
  return availableAxes.end();
}

// Removes `availableAxis` from `availableAxes` and adds the prefix and suffix
// of `availableAxis` that don't overlap with `overlap` back to `availableAxes`.
//
// We assume that `availableAxis` overlaps with `overlap`.
void removeOverlapFromAvailable(AxisRefAttr availableAxis, AxisRefAttr overlap,
                                AvailableAxes& availableAxes, MeshAttr mesh) {
  availableAxes.erase(availableAxis);
  if (OptionalAxisRef prefix = availableAxis.getPrefixWithoutOverlap(overlap)) {
    availableAxes.insert(*prefix);
  }
  if (OptionalAxisRef suffix =
          availableAxis.getSuffixWithoutOverlap(overlap, mesh)) {
    availableAxes.insert(*suffix);
  }
}

// Adds `axis` to `availableAxes` and merges it with sub-axes in
// `availableAxes` that can be merged with `axis`.
//
// We assume that `axis` doesn't overlap with any axis in `availableAxes`.
void addAvailableAxis(AxisRefAttr axis, AvailableAxes& availableAxes,
                      MeshAttr mesh) {
  // `lower_bound` returns the first available axis greater or equal to `axis`,
  // and we know `axis` doesn't overlap with any available axis.
  auto afterIt = availableAxes.lower_bound(axis);
  auto beforeIt = getPrevOrEnd(afterIt, availableAxes);
  AxisRefAttr axisToAdd = axis;
  // Try to merge `axisToAdd` with the first axis greater than it from the left.
  if (afterIt != availableAxes.end() && axisToAdd.canMerge(*afterIt)) {
    axisToAdd = axisToAdd.merge(*afterIt, mesh);
    availableAxes.erase(afterIt);
  }

  // Try to merge `axisToAdd` with the last axis less than it from the right.
  if (beforeIt != availableAxes.end() && beforeIt->canMerge(axisToAdd)) {
    axisToAdd = beforeIt->merge(axisToAdd, mesh);
    availableAxes.erase(beforeIt);
  }
  availableAxes.insert(axisToAdd);
}

// If there is a prefix of `axis` that fully overlaps with an axis in
// `availableAxes`, returns that prefix and removes it from `availableAxes`.
// Otherwise, returns `std::nullopt` and leaves `availableAxes` unchanged.
std::optional<AxisRefAttr> takeAvailablePrefix(AxisRefAttr axis,
                                               AvailableAxes& availableAxes,
                                               MeshAttr mesh) {
  // It's enough to check the first overlapping axis since any other overlapping
  // axis would necessarily not fully overlap with a prefix of `axis`.
  auto availableIt = getFirstOverlapping(axis, availableAxes);
  if (availableIt == availableAxes.end()) {
    return std::nullopt;
  }
  AxisRefAttr availableAxis = *availableIt;
  if (OptionalAxisRef result = axis.getPrefixWithOverlap(availableAxis, mesh)) {
    removeOverlapFromAvailable(availableAxis, *result, availableAxes, mesh);
    return result;
  }
  return std::nullopt;
}

// Removes all axis refs in `axes` from `availableAxes`.
//
// We assume for every axis ref in `axes` there is exactly one axis ref in
// `availableAxes` that contains it, and if they aren't equal, we remove the
// containing axis and add back the prefix and suffix that don't overlap, if
// exist.
void removeUnavailableAxes(ArrayRef<AxisRefAttr> axes, MeshAttr mesh,
                           AvailableAxes& availableAxes) {
  for (AxisRefAttr axis : axes) {
    removeOverlapFromAvailable(*getFirstOverlapping(axis, availableAxes), axis,
                               availableAxes, mesh);
  }
}

// Returns all available axes or sub-axes in `mesh` that aren't used in
// `axesPerDim`.
AvailableAxes getAvailableAxes(ArrayRef<SmallVector<AxisRefAttr>> axesPerDim,
                               MeshAttr mesh) {
  AvailableAxes unboundAxes;
  for (MeshAxisAttr axis : mesh.getAxes()) {
    unboundAxes.insert(AxisRefAttr::get(mesh.getContext(), axis.getName()));
  }
  for (ArrayRef<AxisRefAttr> axes : axesPerDim) {
    removeUnavailableAxes(axes, mesh, unboundAxes);
  }
  return unboundAxes;
}

// Returns the axes to slice for a specific dimension.
//
// If `inAxes` is empty, the prefix of `outAxes` that is available (i.e., fully
// contained by axes in `availableAxes`) can be sliced. The slicing axes are
// removed from `outAxes` and `availableAxes`, and added to `currentAxes`.
SmallVector<AxisRefAttr> getSlicingAxes(const AxisList& inAxes,
                                        AxisList& outAxes,
                                        SmallVector<AxisRefAttr>& currentAxes,
                                        AvailableAxes& availableAxes,
                                        MeshAttr mesh) {
  if (!inAxes.empty()) {
    return {};
  }
  SmallVector<AxisRefAttr> slicingAxes;
  while (!outAxes.empty()) {
    AxisRefAttr outAxis = outAxes.front();
    std::optional<AxisRefAttr> availablePrefix =
        takeAvailablePrefix(outAxis, availableAxes, mesh);
    if (!availablePrefix) {
      break;
    }
    slicingAxes.push_back(*availablePrefix);
    addAxisOrMerge(currentAxes, *availablePrefix, mesh);
    outAxes.pop_front();
    if (*availablePrefix != outAxis) {
      // Safe to dereference since we know `availablePrefix` and `outAxis` have
      // a common prefix and aren't equal.
      outAxes.push_front(
          *outAxis.getSuffixWithoutOverlap(*availablePrefix, mesh));
      break;
    }
  }
  return slicingAxes;
}

// Returns the axes to gather for a specific dimension.
//
// All axes in `inAxes` are gathered greedily. The gathering axes are removed
// from `availableAxes`, popped from the back of `currentAxes`, and `inAxes` is
// cleared.
SmallVector<AxisRefAttr> getGatheringAxes(AxisList& inAxes,
                                          SmallVector<AxisRefAttr>& currentAxes,
                                          AvailableAxes& availableAxes,
                                          MeshAttr mesh) {
  if (inAxes.empty()) {
    return {};
  }
  SmallVector<AxisRefAttr> gatheringAxes = llvm::to_vector(inAxes);
  currentAxes.pop_back_n(inAxes.size() - 1);
  if (OptionalAxisRef prefix =
          currentAxes.back().getPrefixWithoutOverlap(inAxes.front())) {
    currentAxes.back() = *prefix;
  } else {
    currentAxes.pop_back();
  }

  for (AxisRefAttr axis : inAxes) {
    addAvailableAxis(axis, availableAxes, mesh);
  }
  inAxes.clear();
  return gatheringAxes;
}

class ReshardPattern : public OpConversionPattern<ReshardOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  // For the moment we only consider all_gather and all_slice.
  // TODO(b/380226848): Add support for other collectives.
  LogicalResult matchAndRewrite(
      ReshardOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    TensorShardingAttr inputSharding = getSharding(adaptor.getInput());
    TensorShardingAttr outputSharding = adaptor.getSharding();
    // Here it's safe to assume that shardings' meshes have a name.
    if (inputSharding.getRank() != outputSharding.getRank() ||
        inputSharding.getMeshName() != outputSharding.getMeshName()) {
      return rewriter.notifyMatchFailure(
          op, [](Diagnostic& diag) { diag << "Incompatible shardings"; });
    }
    int64_t rank = inputSharding.getRank();

    // TODO(tomnatan): we should verify that the operand of ReshardOp has a
    // sharding.
    // TODO(tomnatan): use a SymbolTable.

    MeshAttr mesh = inputSharding.getMesh(op);
    SmallVector<AxisList> inAxesPerDim = getAxesPerDim<AxisList>(inputSharding);
    SmallVector<AxisList> outAxesPerDim =
        getAxesPerDim<AxisList>(outputSharding);
    // We remove the common prefix of `inAxes` and `outAxes`, since those axes
    // stay exactly the same during the reshard. We are left with `inAxes` that
    // need to be transformed into `outAxes`, via a sequence of collectives.
    for (auto [inAxes, outAxes] :
         llvm::zip_equal(inAxesPerDim, outAxesPerDim)) {
      removeCommonPrefix(inAxes, outAxes, mesh);
    }

    auto hasRemainingAxes = [](const AxisList& axes) { return !axes.empty(); };
    bool hasRemainingInAxes = llvm::any_of(inAxesPerDim, hasRemainingAxes);
    bool hasRemainingOutAxes = llvm::any_of(outAxesPerDim, hasRemainingAxes);

    if (!hasRemainingInAxes && !hasRemainingOutAxes) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    SmallVector<SmallVector<AxisRefAttr>> currentAxesPerDim =
        getAxesPerDim<SmallVector<AxisRefAttr>>(inputSharding);
    AvailableAxes availableAxes = getAvailableAxes(currentAxesPerDim, mesh);

    Value input = adaptor.getInput();
    MLIRContext* context = rewriter.getContext();

    auto getCurrentSharding = [&]() {
      return TensorShardingAttr::getClosed(
          context, inputSharding.getMeshOrRef(), currentAxesPerDim);
    };

    SmallVector<AxisRefListAttr> collectiveAxesPerDim(rank);

    // We aren't done until both `inAxesPerDim` and `outAxesPerDim` are
    // empty.
    // TODO(b/380226848): this is an initial implementation that only inserts
    // all-gathers and all-slices, and greedily all-gathers axes after the first
    // attempt to insert an all-slice.
    while (hasRemainingInAxes || hasRemainingOutAxes) {
      // 1. Try to insert an all-slice first, as it decreases the size of the
      // tensor.
      hasRemainingOutAxes = false;
      bool hasSlicingAxes = false;
      for (auto [inAxes, outAxes, currentAxes, collectiveAxes] :
           llvm::zip_equal(inAxesPerDim, outAxesPerDim, currentAxesPerDim,
                           collectiveAxesPerDim)) {
        SmallVector<AxisRefAttr> slicingAxes =
            getSlicingAxes(inAxes, outAxes, currentAxes, availableAxes, mesh);
        if (!slicingAxes.empty()) {
          hasSlicingAxes = true;
        }
        if (!outAxes.empty()) {
          hasRemainingOutAxes = true;
        }
        collectiveAxes = AxisRefListAttr::get(context, slicingAxes);
      }
      if (hasSlicingAxes) {
        input = rewriter.create<AllSliceOp>(
            op.getLoc(), input, collectiveAxesPerDim, getCurrentSharding());
      }

      // 2. Try to insert an all-gather, that increases the size of the tensor.
      hasRemainingInAxes = false;
      bool hasGatheringAxes = false;
      for (auto [inAxes, currentAxes, collectiveAxes] : llvm::zip_equal(
               inAxesPerDim, currentAxesPerDim, collectiveAxesPerDim)) {
        SmallVector<AxisRefAttr> gatheringAxes =
            getGatheringAxes(inAxes, currentAxes, availableAxes, mesh);
        if (!gatheringAxes.empty()) {
          hasGatheringAxes = true;
        }
        collectiveAxes = AxisRefListAttr::get(context, gatheringAxes);
      }
      if (hasGatheringAxes) {
        input = rewriter.create<AllGatherOp>(
            op.getLoc(), input, collectiveAxesPerDim, getCurrentSharding());
      }
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};

struct ReshardToCollectivesPass
    : public impl::ReshardToCollectivesPassBase<ReshardToCollectivesPass> {
  using ReshardToCollectivesPassBase::ReshardToCollectivesPassBase;

  LogicalResult initialize(MLIRContext* context) final {
    target = std::make_shared<ConversionTarget>(*context);
    target->addIllegalOp<ReshardOp>();
    target->addLegalOp<AllGatherOp, AllSliceOp>();

    RewritePatternSet patternsInternal(context);
    patternsInternal.add<ReshardPattern>(context);
    patterns = std::move(patternsInternal);

    return success();
  }

  void runOnOperation() final {
    if (failed(applyPartialConversion(getOperation(), *target, patterns))) {
      signalPassFailure();
    }
  }

 private:
  std::shared_ptr<ConversionTarget> target;
  FrozenRewritePatternSet patterns;
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
