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

#ifndef SHARDY_DIALECT_SDY_IR_UTILS_H_
#define SHARDY_DIALECT_SDY_IR_UTILS_H_

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

template <typename... Ts>
void unreachableFormatv(const char* format, Ts&&... vals) {
  llvm_unreachable(
      llvm::formatv(format, std::forward<Ts>(vals)...).str().c_str());
}

template <class Dialect>
bool inDialect(Operation* op) {
  return op->getDialect()->getNamespace() == Dialect::getDialectNamespace();
}

// Emits a warning once for the given `flag`, with `op` attached as a note
// if `MLIRContext::shouldPrintOpOnDiagnostic` is true (assuming the op is
// verified).
void emitOpWarningOnce(llvm::once_flag& flag, Operation* op, StringRef msg);

// Converts `attr` to string.
std::string attributeToString(Attribute attr);

// Converts `op` to string with location information.
std::string operationToString(Operation* op);

// Converts `value` to string with location information.
std::string valueToString(Value value);

// If the given `type` is a `ShapedType` with a static shape, returns it,
// otherwise returns nullptr.
ShapedType dynCastStaticShapedType(Type type);

// Returns true if the given `type` is a `ShapedType` with a static shape.
bool isStaticShapedType(Type type);

// Returns the shape of the given `value` if its type is a `ShapeTensor`,
// otherwise returns an empty array.
//
// Assumes the `ShapeTensor` has a rank.
ArrayRef<int64_t> getTensorShape(Value value);

// Returns the rank of the given `type` if it is a `ShapedType`, otherwise
// returns 0.
//
// Assumes the `ShapedType` has a rank.
int64_t getTensorRank(Type type);

// Returns the rank of the given `value` if its type is a `ShapeTensor`,
// otherwise returns 0.
//
// Assumes the `ShapeTensor` has a rank.
int64_t getTensorRank(Value value);

// Returns true if the value is a tensor with rank 0.
int64_t isScalar(Value value);

// Returns the product of the given mesh axis sizes.
int64_t getTotalAxesSize(ArrayRef<MeshAxisAttr> axes);

// Returns the product of the given axis sizes from the given mesh.
int64_t getTotalAxesSize(ArrayRef<AxisRefAttr> axes, MeshAttr mesh);

// Looks up the mesh symbol with the given `meshName` in `symbolTable`, and
// returns it if it exists in the table, or nullptr otherwise.
MeshOp getMeshOp(const SymbolTable& symbolTable, StringRef meshName);

// Looks up the mesh symbol with the given `meshSymName` in the symbol table of
// the enclosing module of `op`, and returns it if it exists in the table, or
// nullptr otherwise.
MeshOp getMeshOp(Operation* op, SymbolRefAttr meshSymName);

// Looks up the mesh symbol with the given `meshName` in the symbol table of
// the enclosing module of `op`, and returns it if it exists in the table, or
// nullptr otherwise.
MeshOp getMeshOp(Operation* op, StringRef meshName);

// If `meshOrRef` is a `MeshAttr`, returns it, otherwise, looks up the
// referenced mesh symbol in `symbolTable`, and returns its `MeshAttr`
// if it exists in the table, or nullptr otherwise.
MeshAttr getMeshOrLookup(const SymbolTable& symbolTable, Attribute meshOrRef);

// If `meshOrRef` is a `MeshAttr`, returns it, otherwise, looks up the
// referenced mesh symbol in the symbol table of the enclosing module of `op`,
// and returns its `MeshAttr` if it exists in the table, or nullptr otherwise.
MeshAttr getMeshOrLookup(Operation* op, Attribute meshOrRef);

// Looks up the mesh symbol with the given `meshName` in `symbolTable`, and
// returns its `MeshAttr` if it exists in the table, or nullptr otherwise.
MeshAttr getMeshAttr(const SymbolTable& symbolTable, StringRef meshName);

// Looks up the mesh symbol with the given `meshSymName` in `symbolTable`, and
// returns its `MeshAttr` if it exists in the table, or nullptr otherwise.
MeshAttr getMeshAttr(const SymbolTable& symbolTable, SymbolRefAttr meshSymName);

// Looks up the mesh symbol with the given `meshName` in the symbol table of
// the enclosing module of `op`, and returns its `MeshAttr` if it exists in the
// table, or nullptr otherwise.
MeshAttr getMeshAttr(Operation* op, StringRef meshName);

// Looks up the mesh symbol with the given `meshSymName` in the symbol table of
// the enclosing module of `op`, and returns its `MeshAttr` if it exists in the
// table, or nullptr otherwise.
MeshAttr getMeshAttr(Operation* op, SymbolRefAttr meshSymName);

// Returns the common mesh (or a reference to it) bound by all the
// `TensorShardingAttr`s or nullptr if there is none.
//
// Ignores empty meshes unless all meshes are empty.
//
// If there is a common mesh, returns the first inlined common mesh or reference
// to it encountered.
//
// If `ignoreDeviceIds` is true, returns the common mesh by ignoring the
// device order. I.e., it returns a mesh, that is common to all shardings on
// axes, and the device order is arbitrary, except if one of the meshes has iota
// device order, the device order is iota.
Attribute getCommonMeshOrRef(ArrayRef<TensorShardingAttr> operandShardings,
                             ArrayRef<TensorShardingAttr> resultsShardings,
                             const SymbolTable& symbolTable,
                             bool ignoreDeviceIds = false);

// Returns the common `MeshAttr` bound by all the `TensorShardingAttr`s or
// nullptr if there is none.
//
// Ignores empty meshes unless all meshes are empty.
MeshAttr getCommonMesh(ArrayRef<TensorShardingAttr> shardings,
                       const SymbolTable& symbolTable);

// Returns the common `MeshAttr` bound by all the `TensorShardingAttr`s or
// nullptr if there is none.
//
// Ignores empty meshes unless all meshes are empty.
MeshAttr getCommonMesh(ArrayRef<TensorShardingAttr> operandShardings,
                       ArrayRef<TensorShardingAttr> resultsShardings,
                       const SymbolTable& symbolTable);

// Returns the common `MeshAttr` bound by all the `TensorShardingAttr`s or
// nullptr if there is none.
//
// Ignores empty meshes unless all meshes are empty.
MeshAttr getCommonMesh(ArrayRef<TensorShardingAttr> operandShardings,
                       ArrayRef<TensorShardingAttr> resultsShardings,
                       Operation* op);

// Returns the name of the common mesh referenced by all the
// `TensorShardingAttr`s or std::nullopt if there is none.
//
// Ignores empty meshes unless all meshes are empty, and assumes there are no
// inlined meshes and no two mesh names refer to the same `MeshAttr` (otherwise
// one of will be returned arbitrarily).
//
// See the documentation of `getCommonMeshOrRef` for the use of
// `ignoreDeviceIds`.
std::optional<StringRef> getCommonMeshName(
    ArrayRef<TensorShardingAttr> operandShardings,
    ArrayRef<TensorShardingAttr> resultsShardings,
    const SymbolTable& symbolTable, bool ignoreDeviceIds);

// Creates the symbol equivalent of a factor index:
//   -  0 -> 'i'
//   -  1 -> 'j'
//   - 17 -> 'z'
//   - 18 -> 'z_1'
std::string factorSymbolString(int64_t factor);

// Returns the string representation of `attr` without dialect wrapping
//
// If `stripMnemonic` is true, also strips the mnemonic of the attribute.
template <class AttrTy>
std::string strippedAttrString(AttrTy attr, bool stripMnemonic = false) {
  std::string result;
  llvm::raw_string_ostream os(result);
  attr.printStripped(os);
  if (stripMnemonic) {
    result.erase(0, attr.getMnemonic().size());
  }
  return result;
}

// Returns the string representation of an `ArrayRef` of attributes, `attrs`,
// without dialect wrapping.
//
// If `stripMnemonic` is true, also strips the mnemonic of the attribute.
template <class AttrTy>
std::string strippedAttrsString(ArrayRef<AttrTy> attrs,
                                bool stripMnemonic = false) {
  std::string result = "[";
  llvm::raw_string_ostream os(result);
  llvm::interleaveComma(attrs, os, [&](AttrTy attr) {
    os << strippedAttrString(attr, stripMnemonic);
  });
  result += "]";
  return result;
}

template <class Container>
void addAxisOrMerge(Container& container, AxisRefAttr axis, MeshAttr mesh) {
  if (!container.empty() && container.back().canMerge(axis)) {
    container.back() = container.back().merge(axis, mesh);
  } else {
    container.push_back(axis);
  }
}

// Sorts `axes` by `mesh`s comparator, and merges axes that can be merged.
// Assumes that there is no overlap between elements in `axes`.
void sortAndMergeAxes(SmallVector<AxisRefAttr>& axes, MeshAttr mesh);

// Returns the defining op of `value`, if it's an op result, or the parent op
// if it's a block argument.
Operation* getOwningOp(Value value);

// Returns the given `value`, if a sharding can be attached to it, or another
// value that holds the sharding for `value` (e.g. the operand corresponding to
// a block argument in a control flow op).
//
// Some op results and block arguments don't have shardings attached to them.
// Instead we recursively loop through the defining op of these ops' operands.
//
// Returns an empty value if the given `value` has no shardable value, e.g., a
// scalar block argument of a reduction function.
Value getShardableValue(Value value);

// Returns the sharding of the given `value`, whose location depends on the type
// of the value.
//
// For example, the sharding of a function block argument is a function argument
// attr.
//
// Some op results and block arguments don't have shardings attached to them.
// Instead we recursively loop through the defining op of these ops' operands.
//
// Returns an empty `TensorShardingAttr` if the given `value` has no sharding or
// if it has no shardable value (see `getShardableValue`)
//
// If `removeManualAxes` is true, then manual axes are removed from the returned
// sharding if `value` is a block argument of a `ManualComputationOp`.
TensorShardingAttr getSharding(Value value);

// Returns the sharding of the given `value`, or a fully open (closed) empty
// `TensorShardingAttr` if `value` doesn't have a sharding and `closedIfMissing`
// is false (true).
TensorShardingAttr getOrCreateSharding(Value value, StringRef meshName,
                                       bool closedIfMissing = false);

// Returns the sharding of the given `value`, or a fully open (closed) empty
// `TensorShardingAttr` if `value` doesn't have a sharding and `closedIfMissing`
// is false (true).
TensorShardingAttr getOrCreateSharding(Value value, Attribute meshOrRef,
                                       bool closedIfMissing = false);

// Sets the `TensorShardingPerValueAttr` of the given `op`, but
// replaces the sharding at the given `index` with the given `sharding`.
//
// If no `TensorShardingPerValueAttr` exists, it's set to a new one with
// a fully open sharding for each result with the same mesh name as `sharding`.
void replaceShardingAtIndex(Operation* op, unsigned index,
                            TensorShardingAttr sharding);

// Sets the sharding of the given `value`, whose location depends on the type of
// the value, to `sharding`.
//
// For example, the sharding of a function block argument is a function argument
// attr.
//
// Some op results and block arguments don't have shardings attached to them.
// Instead we recursively loop through the defining op of these ops' operands.
//
// If `addManualAxes` is true, then the manual axes are added to the given
// `sharding` if `value` is a block argument of a `ManualComputationOp`.
void setSharding(Value value, TensorShardingAttr sharding);

// Return the sharding of the `resNum` result of the given `funcOp`.
TensorShardingAttr getFuncResultSharding(func::FuncOp funcOp, int64_t resNum);

// Sets the sharding of the `resNum` result of the given `funcOp` to `sharding`.
void setFuncResultSharding(func::FuncOp funcOp, int64_t resNum,
                           TensorShardingAttr sharding);

// Returns the sharding of each value in `values`. Returns an empty array if the
// op has no sharding attributes.
SmallVector<TensorShardingAttr> getShardings(ValueRange values);

// Gets the sharding attributes that live on `op` inside the `sdy.sharding`
// attr. Returns an empty array if the op has no sharding attributes.
ArrayRef<TensorShardingAttr> getShardings(Operation* op);

// Gets the `TensorShardingPerValueAttr` that lives on `op` inside the
// `sdy.sharding` attr. Returns a null attr if the op has no such attr.
TensorShardingPerValueAttr getShardingPerValue(Operation* op);

// Sets the sharding attributes on `op` to a
// `TensorShardingPerValueAttr` named `sdy.sharding` consisting of `shardings`.
void setShardings(Operation* op, ArrayRef<TensorShardingAttr> shardings);

// Sets the sharding attributes on `op` to a
// `TensorShardingPerValueAttr` named `sdy.sharding` consisting of `shardings`.
void setShardings(Operation* op, TensorShardingPerValueAttr shardingPerValue);

// Cleanup the module by removing sharding rule attrs. Keep any user specified
// ones.
void removeShardingRules(Operation* rootOp);

// Gets the `op` body's terminator.
template <typename RegionOpTy>
Operation* getBodyTerminator(RegionOpTy op) {
  return op.getBody().front().getTerminator();
}

// Gets the operands of the `op` body terminator.
template <typename RegionOpTy>
MutableArrayRef<OpOperand> getBodyTerminatorOpOperands(RegionOpTy op) {
  return getBodyTerminator(op)->getOpOperands();
}

// Gets the values returned from the `op` body terminator.
template <typename RegionOpTy>
ValueRange getBodyTerminatorOperands(RegionOpTy op) {
  return getBodyTerminator(op)->getOperands();
}

// Gets the value of the `op` body terminator at `index`.
template <typename RegionOpTy>
Value getBodyTerminatorOperand(RegionOpTy op, int64_t index) {
  return getBodyTerminator(op)->getOperand(index);
}

// Gets the value types returned from the `op` body terminator.
template <typename RegionOpTy>
TypeRange getBodyTerminatorOpOperandTypes(RegionOpTy op) {
  return getBodyTerminator(op)->getOperandTypes();
}

// Returns the greatest common prefix of given two arrays axis refs.
SmallVector<AxisRefAttr> getGreatestCommonPrefix(ArrayRef<AxisRefAttr> first,
                                                 ArrayRef<AxisRefAttr> second);

// Inlines (i.e., move) operations from region `src` into `dst` and converts the
// terminator of each block in `dst` to `TerminatorOpTy`. The `rewriter`'s
// insertion point is modified.
template <typename TerminatorOpTy>
void inlineRegionAndConvertTerminatorOp(Region& src, Region& dst,
                                        PatternRewriter& rewriter) {
  rewriter.inlineRegionBefore(src, dst, dst.begin());

  for (Block& block : dst.getBlocks()) {
    Operation* returnOp = block.getTerminator();

    rewriter.setInsertionPointAfter(returnOp);
    rewriter.replaceOpWithNewOp<TerminatorOpTy>(returnOp,
                                                returnOp->getOperands());
  }
}

// Inlines (i.e., move) operations from region `src` into `dst` and converts the
// terminator of each block in `dst` to `TerminatorOpTy`.
template <typename TerminatorOpTy>
void inlineRegionAndConvertTerminatorOp(Region& src, Region& dst) {
  dst.takeBody(src);

  for (Block& block : dst.getBlocks()) {
    Operation* returnOp = block.getTerminator();
    OpBuilder::atBlockEnd(&block).create<TerminatorOpTy>(
        returnOp->getLoc(), returnOp->getOperands());
    returnOp->erase();
  }
}

// Clones region `src`, inserts `src` into the start of `dst`, and converts the
// terminator of each block in `dst` to `TerminatorOpTy`.
template <typename TerminatorOpTy>
void cloneRegionAndConvertTerminatorOp(Region& src, Region& dst,
                                       RewriterBase& rewriter) {
  Block::iterator savedInsertionPoint = rewriter.getInsertionPoint();
  Block* savedBlock = rewriter.getInsertionBlock();
  rewriter.cloneRegionBefore(src, dst, dst.begin());

  for (auto& block : dst.getBlocks()) {
    Operation* returnOp = block.getTerminator();
    rewriter.setInsertionPointAfter(returnOp);
    rewriter.replaceOpWithNewOp<TerminatorOpTy>(returnOp,
                                                returnOp->getOperands());
  }

  rewriter.setInsertionPoint(savedBlock, savedInsertionPoint);
}

// Clones region `src`, inserts `src` into the start of `dst`, and converts the
// terminator of each block in `dst` to `TerminatorOpTy`.
template <typename TerminatorOpTy>
void cloneRegionAndConvertTerminatorOp(Region& src, Region& dst) {
  IRRewriter rewriter(src.getContext());
  cloneRegionAndConvertTerminatorOp<TerminatorOpTy>(src, dst, rewriter);
}

// Gets the enclosing `OpTy` of the given `op`. If the `op` is already of type
// `OpTy`, returns it.
template <class OpTy>
OpTy getEnclosingOfType(Operation* op) {
  if (auto typedOp = mlir::dyn_cast<OpTy>(op)) {
    return typedOp;
  }
  return op->getParentOfType<OpTy>();
}

// Builds an open `TensorSharding` for each type in `types`.
SmallVector<TensorShardingAttr> getFullyOpenShardings(MLIRContext* context,
                                                      TypeRange types,
                                                      StringRef meshName);

// Builds an closed `TensorSharding` for each type in `types`.
SmallVector<TensorShardingAttr> getFullyClosedShardings(MLIRContext* context,
                                                        TypeRange types,
                                                        StringRef meshName);

// Builds an open `TensorSharding` for each type in `types`, but
// with the sharding at `index` replaced with `sharding`.
SmallVector<TensorShardingAttr> getOpenShardingsWithShardingAtIndex(
    MLIRContext* context, TypeRange types, int64_t index,
    TensorShardingAttr sharding);

// Removes manual axes from the sharding.
//
// Guaranteed by verification that all in/out shardings in a
// `ManualComputationOp` are prefixed with the manual axes. So this removes the
// prefix of manual axes (if any exist) from each dim sharding.
TensorShardingAttr eraseManualAxes(TensorShardingAttr outerManualSharding,
                                   ArrayRef<StringAttr> manualAxes);

// Removes free axes from the sharding.
//
// Guaranteed by verification that all in/out shardings in a
// `ManualComputationOp` are prefixed with the manual axes. So this removes the
// suffix of free axes (if any exist) from each dim sharding.
TensorShardingAttr eraseFreeAxes(TensorShardingAttr outerManualSharding,
                                 ArrayRef<StringAttr> manualAxes);

// `dimAxes` is assumed to be the concatenation of some manual axes and some
// free axes. This returns an iterator to the first free axis in `dimAxes`.
//
// Axes in the range [0, firstFreeAxis) are manual axes, and
// [firstFreeAxis, dimAxes.size()) are free axes.
ArrayRef<AxisRefAttr>::const_iterator getFirstFreeAxisIter(
    ArrayRef<AxisRefAttr> dimAxes, ArrayRef<StringAttr> manualAxes);

// Returns all axes or sub-axes in `axesA` that are not in `axesB`.
//
// Both `axesA` and `axesB` don't need to be sorted. If an axis A in `axesA`
// overlaps with an axis B in `axesB`, the prefix and suffix of A that don't
// overlap with B are added to the result.
//
// The order of axes in `axesA` is preserved in the returned vector.
SmallVector<AxisRefAttr> getAxisSetDiff(ArrayRef<AxisRefAttr> axesA,
                                        ArrayRef<AxisRefAttr> axesB,
                                        MeshAttr mesh);

// Returns true if any of `values` is used by any op of the specified types.
template <class... OpTys>
bool hasAnyUserOfType(ValueRange values) {
  for (Value value : values) {
    return llvm::any_of(value.getUsers(), [](Operation* user) {
      return mlir::isa<OpTys...>(user);
    });
  }
  return false;
}

// Returns true if `op` is only used by ops of the specified types.
template <class... OpTys>
bool hasOnlyUsersOfType(Operation* op) {
  return llvm::all_of(op->getUsers(), [](Operation* user) {
    return mlir::isa<OpTys...>(user);
  });
}

// Helper class to insert or merge `AxisRefAttr`s into a `SmallVector`.
// It implements the interface required by `llvm::transform` to be used as an
// output iterator.
class AddAxisOrMergeInserter {
 public:
  using iterator_category = std::output_iterator_tag;
  using value_type = void;
  using difference_type = void;
  using pointer = void;
  using reference = void;

  explicit AddAxisOrMergeInserter(SmallVector<AxisRefAttr>* newAxisRefs,
                                  const MeshAttr* mesh)
      : axisRefs(newAxisRefs), mesh(mesh) {}

  AddAxisOrMergeInserter& operator=(AxisRefAttr value) {
    sdy::addAxisOrMerge(*axisRefs, value, *mesh);
    return *this;
  }

  AddAxisOrMergeInserter& operator=(ArrayRef<AxisRefAttr> values) {
    for (AxisRefAttr value : values) {
      *this = value;
    }
    return *this;
  }

  AddAxisOrMergeInserter& operator*() { return *this; }
  AddAxisOrMergeInserter& operator++() { return *this; }
  AddAxisOrMergeInserter& operator++(int) { return *this; }

 private:
  // Use pointers so that callers like `llvm::transform` can copy the inserter.
  SmallVector<AxisRefAttr>* axisRefs;
  const MeshAttr* mesh;
};

// Returns true if `op` is used by any op of the specified types.
template <class... OpTys>
bool hasAnyUserOfType(Operation* op) {
  return hasAnyUserOfType(op->getResults());
}

// Returns true if `value` is used by `user`.
bool isUsedBy(Value value, Operation* user);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_IR_UTILS_H_
