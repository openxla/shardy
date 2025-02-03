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
#include <functional>
#include <iterator>
#include <list>
#include <memory>  // IWYU pragma: keep
#include <optional>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
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

using AxesPerDim = SmallVector<SmallVector<AxisRefAttr>>;

// We use an std::list so we can pop from the front and the back and with an
// iterator at constant time.
using AxisList = std::list<AxisRefAttr>;

using AxisRefToDimMap = llvm::SmallDenseMap<AxisRefAttr, int64_t>;

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

// Returns an iterator to the first axis in `orderedAxes` that overlaps with
// `axis`, or `orderedAxes.end()` if there is no such axis.
ArrayRef<AxisRefAttr>::iterator getFirstOverlapping(
    AxisRefAttr axis, ArrayRef<AxisRefAttr> orderedAxes) {
  if (orderedAxes.empty()) {
    return orderedAxes.end();
  }
  auto* afterIt = llvm::lower_bound(orderedAxes, axis);
  // If there is at least one overlapping axis, the first one is necessarily
  // `afterIt` or `beforeIt = std::prev(afterIt)`.
  //
  // Proof:
  // Given the definition of `lower_bound`, we have `beforeIt < A <= afterIt`.
  //
  // - For any entry B with `B < beforeIt < A`, B and `beforeIt` cannot overlap.
  //   Thus `beforeIt` isolates A and B such that they cannot overlap.
  // - For any entry C with `A <= afterIt < C`, `afterIt` and C, if A and C
  //   overlap, then A and `afterIt` must overlap as well.

  if (afterIt != orderedAxes.begin() && std::prev(afterIt)->overlaps(axis)) {
    return std::prev(afterIt);
  }
  if (afterIt != orderedAxes.end() && afterIt->overlaps(axis)) {
    return afterIt;
  }
  return orderedAxes.end();
}

// Returns a map from `AxisRefAttr` to the dimension and index within the
// dimension sharding in `axesPerDim` that this axis appears.
AxisRefToDimMap getAxisRefToDimMap(ArrayRef<AxisList> axesPerDim) {
  AxisRefToDimMap result;
  for (auto [dim, axes] : llvm::enumerate(axesPerDim)) {
    for (AxisRefAttr axis : axes) {
      result.try_emplace(axis, dim);
    }
  }
  return result;
}

SmallVector<AxisRefAttr> getOrderedAxes(ArrayRef<AxisList> axesPerDim) {
  SmallVector<AxisRefAttr> result;
  for (const AxisList& axes : axesPerDim) {
    result.append(axes.begin(), axes.end());
  }
  llvm::sort(result);
  return result;
}

// Remove the common prefix of `inAxesPerDim` and `outAxesPerDim`.
void removeCommonPrefix(SmallVector<AxisList>& inAxesPerDim,
                        SmallVector<AxisList>& outAxesPerDim) {
  for (auto [inAxes, outAxes] : llvm::zip_equal(inAxesPerDim, outAxesPerDim)) {
    while (!inAxes.empty() && !outAxes.empty() &&
           inAxes.front() == outAxes.front()) {
      inAxes.pop_front();
      outAxes.pop_front();
    }
  }
}

// In case an axis A in `axes` overlaps but isn't equal to an axis B in
// `orderedOtherAxes`, decomposes A into 1-3 sub-axes (overlap and
// non-overlapping prefix and suffix), and replaces A with the decomposed
// sub-axes that form it.
void alignSubAxesByDecomposition(AxisList& axes,
                                 ArrayRef<AxisRefAttr> orderedOtherAxes,
                                 MeshAttr mesh) {
  auto axisIt = axes.begin();
  while (axisIt != axes.end()) {
    AxisRefAttr axis = *axisIt;
    auto* overlapIt = getFirstOverlapping(axis, orderedOtherAxes);
    // There are two paths to complete the while loop below:
    // 1. the while condition is not met from the start, in which case we need
    //    to advance `axisIt`.
    // 2. we enter the while until the condition isn't met, in which case we
    //    only need to advance `axisIt` if it points to a created suffix.
    bool axisAdvancedInWhile = false;
    while (overlapIt != orderedOtherAxes.end() && overlapIt->canCoexist(axis) &&
           !overlapIt->contains(axis) && overlapIt->overlaps(axis)) {
      axisIt = axes.erase(axisIt);
      if (OptionalAxisRef prefix = axis.getPrefixWithoutOverlap(*overlapIt)) {
        axisIt = ++axes.insert(axisIt, *prefix);
      }
      axisIt = ++axes.insert(axisIt, *axis.getOverlap(*overlapIt));
      if (OptionalAxisRef suffix =
              axis.getSuffixWithoutOverlap(*overlapIt, mesh)) {
        // If there is a suffix, that should be the next axis to process.
        axisIt = axes.insert(axisIt, *suffix);
        axis = *suffix;
        ++overlapIt;
        axisAdvancedInWhile = false;
      } else {
        // Otherwise, we're done with the current axis.
        axisAdvancedInWhile = true;
        break;
      }
    }
    if (!axisAdvancedInWhile) {
      ++axisIt;
    }
  }
}

// In case two `AxisRefAttr` in `inAxesPerDim` and `outAxesPerDim` respectively
// overlap but aren't equal, decomposes them into up to three sub-axes (overlap
// and non-overlapping prefix and suffix), and replaces each original axis with
// the decomposed sub-axes that form it (see overload above).
//
// For example, "a":(1)8 and "a":(4)4 are decomposed into "a":(1)4, "a":(4)2,
// and "a":(8)2. Then "a":(1)8 is replaced with ["a":(1)4, "a":(4)2] and
// "a":(4)4 is replaced with ["a":(4)2, "a":(8)2].
void alignSubAxesByDecomposition(SmallVector<AxisList>& inAxesPerDim,
                                 SmallVector<AxisList>& outAxesPerDim,
                                 MeshAttr mesh) {
  SmallVector<AxisRefAttr> orderedInAxes = getOrderedAxes(inAxesPerDim);
  SmallVector<AxisRefAttr> orderedOutAxes = getOrderedAxes(outAxesPerDim);
  for (AxisList& inAxes : inAxesPerDim) {
    alignSubAxesByDecomposition(inAxes, orderedOutAxes, mesh);
  }
  for (AxisList& outAxes : outAxesPerDim) {
    alignSubAxesByDecomposition(outAxes, orderedInAxes, mesh);
  }
}

// Removes the axes in `axesToPop` from the back of `currentAxes`.
//
// Note that `axesToPop` can have decomposed sub-axes of an axis in
// `currentAxes`, which is taken into account.
void popBackFromCurrentAxes(SmallVector<AxisRefAttr>& currentAxes,
                            const AxisList& axesToPop,
                            AxisList::iterator startIt) {
  for (auto it = axesToPop.rbegin(); it != std::make_reverse_iterator(startIt);
       ++it) {
    if (auto prefix = currentAxes.back().getPrefixWithoutOverlap(*it)) {
      currentAxes.back() = *prefix;
    } else {
      currentAxes.pop_back();
    }
  }
}

struct AllToAllInfo {
  SmallVector<AxisRefAttr> axes;
  int64_t tgtDim = -1;
};

// A helper class that transforms an input sharding into an output sharding via
// a sequence of collectives, and holds the current state of the transformation.
class CollectiveInserter {
 public:
  CollectiveInserter(TensorShardingAttr inSharding,
                     TensorShardingAttr outSharding, MeshAttr mesh,
                     Value result, ConversionPatternRewriter& rewriter,
                     Location loc)
      : rewriter(rewriter),
        loc(loc),
        mesh(mesh),
        meshOrRef(inSharding.getMeshOrRef()),
        result(result),
        inAxesPerDim(getAxesPerDim<AxisList>(inSharding)),
        outAxesPerDim(getAxesPerDim<AxisList>(outSharding)),
        currentAxesPerDim(getAxesPerDim<SmallVector<AxisRefAttr>>(inSharding)),
        collectiveAxesPerDim(inSharding.getRank()) {
    // We align sub-axes between the input and output axes, so that we can treat
    // sub-axes like full axes and assume any two sub-axes that overlap are also
    // equal, which allows using them as keys in a hash map.
    alignSubAxesByDecomposition(inAxesPerDim, outAxesPerDim, mesh);
    // We remove the common prefix of `inAxesPerDim` and `outAxesPerDim`, since
    // those axes stay exactly the same during the reshard. We are left with
    // `inAxesPerDim` that need to be transformed into `outAxesPerDim`, via a
    // sequence of collectives.
    removeCommonPrefix(inAxesPerDim, outAxesPerDim);

    inAxisToDimMap = getAxisRefToDimMap(inAxesPerDim);
    outAxisToDimMap = getAxisRefToDimMap(outAxesPerDim);
  }

  // Returns true if the input sharding has been transformed into the output
  // sharding, i.e., both `inAxesPerDim` and `outAxesPerDim` are empty.
  bool isDone() const {
    return llvm::all_of(inAxesPerDim, std::mem_fn(&AxisList::empty)) &&
           llvm::all_of(outAxesPerDim, std::mem_fn(&AxisList::empty));
  }

  MLIRContext* getContext() const { return rewriter.getContext(); }

  int64_t getRank() const { return inAxesPerDim.size(); }

  // Returns the result of the last inserted collective, or the initial input if
  // none have been inserted.
  Value getResult() const { return result; }

  // Tries to insert an `sdy.all_gather`.
  void tryAllGather() {
    bool hasGatheringAxes = false;
    for (auto [dim, collectiveAxes] : llvm::enumerate(collectiveAxesPerDim)) {
      SmallVector<AxisRefAttr> gatheringAxes = getGatheringAxes(dim);
      if (!gatheringAxes.empty()) {
        hasGatheringAxes = true;
      }
      collectiveAxes = AxisRefListAttr::get(getContext(), gatheringAxes);
    }
    if (hasGatheringAxes) {
      result = rewriter.create<AllGatherOp>(loc, result, collectiveAxesPerDim,
                                            getCurrentSharding());
    }
  }

  void tryAllSlice() {
    if (std::optional<AxesPerDim> slicingAxesPerDim =
            getSlicingAxesPerDim(inAxesPerDim, outAxesPerDim, currentAxesPerDim,
                                 inAxisToDimMap, mesh)) {
      for (auto [collectiveAxes, slicingAxes] :
           llvm::zip_equal(collectiveAxesPerDim, *slicingAxesPerDim)) {
        collectiveAxes = AxisRefListAttr::get(getContext(), slicingAxes);
      }
      result = rewriter.create<AllSliceOp>(loc, result, collectiveAxesPerDim,
                                           getCurrentSharding());
    }
  }

  // Tries to insert a sequence of `sdy.all_to_all`s.
  void tryAllToAlls() {
    bool allToAllCreated = false;
    do {
      allToAllCreated = false;
      for (int64_t srcDim = 0; srcDim < getRank(); ++srcDim) {
        auto [allToAllAxes, tgtDim] = getAllToAllAxesAndTgtDim(
            srcDim, inAxesPerDim, outAxesPerDim, currentAxesPerDim,
            inAxisToDimMap, outAxisToDimMap, mesh);
        if (!allToAllAxes.empty()) {
          result =
              rewriter.create<AllToAllOp>(loc, result, srcDim, tgtDim,
                                          allToAllAxes, getCurrentSharding());
          allToAllCreated = true;
        }
      }
    } while (allToAllCreated);
  }

 private:
  TensorShardingAttr getCurrentSharding() const {
    return TensorShardingAttr::getClosed(getContext(), meshOrRef,
                                         currentAxesPerDim);
  }

  // Returns the axes to gather for a specific dimension.
  //
  // All axes in `inAxes` are gathered greedily. The gathering axes are popped
  // from the back of `currentAxes` and `inAxes` is cleared. `inAxisToDimMap` is
  // also updated as needed.
  SmallVector<AxisRefAttr> getGatheringAxes(int64_t dim) {
    AxisList& inAxes = inAxesPerDim[dim];
    if (inAxes.empty()) {
      return {};
    }
    SmallVector<AxisRefAttr>& currentAxes = currentAxesPerDim[dim];
    SmallVector<AxisRefAttr> gatheringAxes;
    gatheringAxes.reserve(inAxes.size());
    popBackFromCurrentAxes(currentAxes, inAxes, inAxes.begin());
    for (AxisRefAttr axis : inAxes) {
      addAxisOrMerge(gatheringAxes, axis, mesh);
      inAxisToDimMap.erase(axis);
    }
    inAxes.clear();
    return gatheringAxes;
  }

  // TODO(b/392952931): currently we are greedily slicing and all-to-all-ing
  // axes even if the destination dimension is too small to accommodate the
  // extra axes. This would introduce padding which is sub-optimal, thus we
  // should only do this if the dimension has enough space left, or slice as
  // much as possible to fill the space.

  // Returns the axes to slice for each dimension.
  //
  // For each dimension d, each axis X in `inAxesPerDim[d]` that isn't present
  // in `inAxisToDimMap` (i.e. available to slice) is sliced as follows:
  // - If the last axis Y before X in `inAxesPerDim[d]` that isn't sliced is
  //   mapped to the same dimension d in `inAxisToDimMap`, or there isn't such
  //   an axis, then X is sliced on that dimension.
  // - Otherwise, X is sliced on the mapped dimension (`inAxisToDimMap[Y]`), so
  //   we can later do an all-to-all on a smaller tensor to move both axes to
  //   the other dimension.
  //
  // The slicing axes are added to `currentAxesPerDim` and each is either
  // removed from `outAxesPerDim[d]` if it's where it need to be, or appended to
  // `inAxesPerDim[d]` otherwise. `inAxisToDimMap` is also updated as needed.
  //
  // Returns std::nullopt if there are no slicing axes in any dimension.
  //
  // For example:
  //
  // Reshard: `[{"u"}, {"y"}, {}]` -> `[{"u", "x"}, {}, {"y", "z", "w"}]`
  //
  // Arguments:
  // - `inAxesPerDim = [[], ["y"], []]`,
  // - `outAxesPerDim = [["x"], [], ["y", "z", "w"]]`
  // - `currentAxesPerDim = [["u"], ["y"], []]`
  // - `inAxisToDimMap = [{"y": 1}]` (assumed to be derived from `inAxesPerDim`)
  //
  // Returns: `[["x"], ["z", "w"], []]`, and updates:
  // - `inAxesPerDim = [[], ["y", "z", "w"], []]`,
  // - `outAxesPerDim = [[], [], ["y", "z", "w"]]`
  // - `currentAxesPerDim = [["u", "x"], ["y", "z", "w"], []]`
  // - `inAxisToDimMap = [{"y": 1}, {"z": 1}, {"w": 1}]`
  std::optional<AxesPerDim> getSlicingAxesPerDim(
      SmallVector<AxisList>& inAxesPerDim, SmallVector<AxisList>& outAxesPerDim,
      AxesPerDim& currentAxesPerDim, AxisRefToDimMap& inAxisToDimMap,
      MeshAttr mesh) {
    AxesPerDim slicingAxesPerDim(currentAxesPerDim.size());

    bool hasSlicingAxes = false;
    for (auto [outDim, outAxes] : llvm::enumerate(outAxesPerDim)) {
      auto outIt = outAxes.begin();
      std::optional<int64_t> lastInDim;
      while (outIt != outAxes.end()) {
        AxisRefAttr outAxis = *outIt;
        if (auto inAxisEntryIt = inAxisToDimMap.find(outAxis);
            inAxisEntryIt != inAxisToDimMap.end()) {
          // Out axis isn't available to slice.
          lastInDim = inAxisEntryIt->second;
          ++outIt;
          continue;
        }
        // We should slice `outAxis` at `lastInDim` if present or `outDim`
        // otherwise.
        hasSlicingAxes = true;
        int64_t slicingDim = lastInDim.value_or(outDim);
        addAxisOrMerge(slicingAxesPerDim[slicingDim], outAxis, mesh);
        addAxisOrMerge(currentAxesPerDim[slicingDim], outAxis, mesh);
        AxisList& inAxes = inAxesPerDim[slicingDim];
        if (inAxes.empty() && outIt == outAxes.begin()) {
          // Slicing axis is where it needs to be.
          outIt = outAxes.erase(outIt);
        } else {
          inAxisToDimMap.try_emplace(outAxis, slicingDim);
          inAxes.push_back(outAxis);
          ++outIt;
        }
      }
    }

    return hasSlicingAxes ? std::make_optional(slicingAxesPerDim)
                          : std::nullopt;
  }

  // Returns the axes and target dimension to all-to-all from `srcDim`.
  //
  // The suffix of axes in `inAxesPerDim[srcDim]` that are mapped to the same
  // dimension in `outAxisToDimMap` are all-to-all-ed with the mapped dimension
  // as the target (tgtDim).
  //
  // The axes are popped from the back of `inAxesPerDim[srcDim]` and
  // `currentAxesPerDim`, and each is either removed from
  // `outAxesPerDim[tgtDim]` if it's where it need to be, or appended to
  // `inAxesPerDim[tgtDim]` otherwise. `inAxisToDimMap` is also updated as
  // needed.
  //
  // For example:
  //
  // Reshard: `[{"w"}, {"x", "y", "z"}, {}]` -> `[{"x"}, {}, {"y", "z"}]`
  //
  // Arguments:
  // - `srcDim = 1`
  // - `inAxesPerDim = [["w"], ["x", "y", "z"], []]`,
  // - `outAxesPerDim = [["x"], [], ["y", "z"]]`
  // - `currentAxesPerDim = [["w"], ["x", "y", "z"], []]`
  // - `inAxisToDimMap = [{"x": 1}, {"y": 1}, {"z": 1}, {"w": 0}]`
  //   (assumed to be derived from `inAxesPerDim`)
  // - `outAxisToDimMap = [{"x": 0}, {"y": 1}, {"z": 1}]`
  //   (assumed to be derived from `outAxesPerDim`)
  //
  // First call returns: `{axes = ["y", "z"], tgtDim = 2}`, and updates:
  // - `inAxesPerDim = [["w"], ["x"], []]`,
  // - `outAxesPerDim = [["x"], [], []]`
  // - `currentAxesPerDim = [["w"], ["x"], ["y", "z"]]`
  // - `inAxisToDimMap = [{"x": 1}, {"w": 0}]`
  //
  // Second call returns: `{axes = ["x"], tgtDim = 0}`, and updates:
  // - `inAxesPerDim = [["w", "x"], [], []]`,
  // - `outAxesPerDim = [["x"], [], []]`
  // - `currentAxesPerDim = [["w", "x"], [], ["y", "z"]]`
  // - `inAxisToDimMap = [{"x": 0}, {"w": 0}]`
  AllToAllInfo getAllToAllAxesAndTgtDim(int64_t srcDim,
                                        SmallVector<AxisList>& inAxesPerDim,
                                        SmallVector<AxisList>& outAxesPerDim,
                                        AxesPerDim& currentAxesPerDim,
                                        AxisRefToDimMap& inAxisToDimMap,
                                        const AxisRefToDimMap& outAxisToDimMap,
                                        MeshAttr mesh) {
    AllToAllInfo result;
    auto& [allToAllAxes, tgtDim] = result;

    AxisList& srcInAxes = inAxesPerDim[srcDim];

    auto axisRevIt = srcInAxes.rbegin();
    int64_t numAxes = 0;
    for (; axisRevIt != srcInAxes.rend(); ++axisRevIt) {
      auto outAxisEntryIt = outAxisToDimMap.find(*axisRevIt);
      if (outAxisEntryIt == outAxisToDimMap.end()) {
        break;
      }
      int64_t outAxisDim = outAxisEntryIt->second;
      if (outAxisDim == srcDim || (tgtDim != -1 && outAxisDim != tgtDim)) {
        break;
      }
      tgtDim = outAxisDim;
      ++numAxes;
    }

    if (tgtDim == -1) {
      // Can't do an all-to-all from `srcDim` to any dimension.
      return result;
    }

    auto startInAxisIt = axisRevIt.base();

    SmallVector<AxisRefAttr>& srcCurrentAxes = currentAxesPerDim[srcDim];
    SmallVector<AxisRefAttr>& tgtCurrentAxes = currentAxesPerDim[tgtDim];
    allToAllAxes.reserve(numAxes);

    popBackFromCurrentAxes(srcCurrentAxes, srcInAxes, startInAxisIt);

    AxisList& tgtInAxes = inAxesPerDim[tgtDim];
    AxisList& tgtOutAxes = outAxesPerDim[tgtDim];
    auto srcInAxisIt = startInAxisIt;
    while (srcInAxisIt != srcInAxes.end()) {
      AxisRefAttr axis = *srcInAxisIt;
      addAxisOrMerge(allToAllAxes, axis, mesh);
      addAxisOrMerge(tgtCurrentAxes, axis, mesh);
      srcInAxisIt = srcInAxes.erase(srcInAxisIt);
      inAxisToDimMap.erase(axis);
      if (tgtInAxes.empty() && tgtOutAxes.front() == axis) {
        tgtOutAxes.pop_front();
          } else {
            tgtInAxes.push_back(axis);
            inAxisToDimMap.try_emplace(axis, tgtDim);
          }
        }

        return result;
      }

  ConversionPatternRewriter& rewriter;
  Location loc;
  MeshAttr mesh;
  Attribute meshOrRef;
  Value result;
  SmallVector<AxisList> inAxesPerDim;
  SmallVector<AxisList> outAxesPerDim;
  AxesPerDim currentAxesPerDim;
  SmallVector<AxisRefListAttr> collectiveAxesPerDim;
  AxisRefToDimMap inAxisToDimMap;
  AxisRefToDimMap outAxisToDimMap;
};

class ReshardPattern : public OpConversionPattern<ReshardOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      ReshardOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    TensorShardingAttr inSharding = getSharding(adaptor.getInput());
    TensorShardingAttr outSharding = adaptor.getSharding();
    // Here it's safe to assume that shardings' meshes have a name.
    if (inSharding.getRank() != outSharding.getRank() ||
        inSharding.getMeshName() != outSharding.getMeshName()) {
      return rewriter.notifyMatchFailure(
          op, [](Diagnostic& diag) { diag << "Incompatible shardings"; });
    }

    // TODO(tomnatan): we should verify that the operand of ReshardOp has a
    // sharding.
    // TODO(tomnatan): use a SymbolTable.

    CollectiveInserter collectiveInserter(
        inSharding, outSharding, inSharding.getMesh(op), adaptor.getInput(),
        rewriter, op.getLoc());

    while (!collectiveInserter.isDone()) {
      // 1. Try to insert an all-slice, that decreases the size of the tensor.
      collectiveInserter.tryAllSlice();

      // 2. Try to insert all-to-alls, that preserves the size of the tensor.
      collectiveInserter.tryAllToAlls();

      // 3. Try to insert an all-gather, that increases the size of the tensor.
      collectiveInserter.tryAllGather();
    }

    rewriter.replaceOp(op, collectiveInserter.getResult());
    return success();
  }
};

struct ReshardToCollectivesPass
    : public impl::ReshardToCollectivesPassBase<ReshardToCollectivesPass> {
  using ReshardToCollectivesPassBase::ReshardToCollectivesPassBase;

  LogicalResult initialize(MLIRContext* context) final {
    target = std::make_shared<ConversionTarget>(*context);
    target->addIllegalOp<ReshardOp>();
    target->addLegalOp<AllGatherOp, AllSliceOp, AllToAllOp>();

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
