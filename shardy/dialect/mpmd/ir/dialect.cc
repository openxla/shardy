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

#include "shardy/dialect/mpmd/ir/dialect.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/InliningUtils.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/fragment_arg_res_attrs.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/parsers.h"
#include "shardy/dialect/sdy/ir/printers.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::mpmd {

using ::mlir::func::FuncOp;
using ::mlir::sdy::TensorShardingAttr;

namespace {

struct MpmdDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // ATM we have two types of calls: FragmentCalls and CallOps. It is never
  // legal to inline the former. It is legal to inline the latter (if late
  // enough in the compiler pipeline), but when we do so, we want to make sure
  // we copy the call_counter attributes. So we will use our own inline pass.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return false;
  }
  // MPMD region-based ops include fragments and control-flow loops, and they
  // can always be the destination of an inlined call.
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       IRMapping& valueMapping) const final {
    return true;
  }
  // Operations in mpmd dialect are legal to inline since they are pure.
  bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final {
    return true;
  }
};

}  // namespace

void MpmdDialect::initialize() {
  addInterface<MpmdDialectInlinerInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "shardy/dialect/mpmd/ir/attrs.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "shardy/dialect/mpmd/ir/types.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "shardy/dialect/mpmd/ir/ops.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TopologyAttr
//===----------------------------------------------------------------------===//

LogicalResult TopologyAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ArrayRef<NamedMeshAttr> meshes) {
  if (meshes.empty()) {
    return emitError() << "TopologyAttr must have at least one mesh.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MeshTensorType
//===----------------------------------------------------------------------===//

namespace {

StringAttr ParseMemoryKind(AsmParser& parser) {
  StringAttr memory_kind;
  // We always parse > since the memory)_kind is always the last attribute of
  // the type.
  if (parser.parseKeyword("memory_kind") || parser.parseEqual() ||
      parser.parseAttribute(memory_kind) || parser.parseGreater()) {
    return {};
  }
  return memory_kind;
}

}  // namespace

Type MeshTensorType::parse(AsmParser& parser) {
  StringAttr mesh_name_attr;
  RankedTensorType ranked_tensor_type;
  if (parser.parseLess() || parser.parseAttribute(mesh_name_attr) ||
      parser.parseComma() || parser.parseType(ranked_tensor_type)) {
    return {};
  }

  if (!parser.parseOptionalGreater()) {
    return MeshTensorType::get(parser.getContext(), mesh_name_attr.getValue(),
                               ranked_tensor_type);
  }

  if (parser.parseComma()) {
    return {};
  }

  sdy::TensorShardingAttr sharding;

  if (!parser.parseOptionalKeyword("sharding")) {
    // Must have a sharding.
    if (parser.parseEqual() ||
        parser.parseCustomAttributeWithFallback(sharding)) {
      return {};
    }
  } else {
    // If there's no sharding, we must have a memory kind.
    if (StringAttr memory_kind = ParseMemoryKind(parser)) {
      return MeshTensorType::get(parser.getContext(), mesh_name_attr.getValue(),
                                 ranked_tensor_type, sharding, memory_kind);
    }
    return {};
  }

  if (!parser.parseOptionalComma()) {
    // If there's another comma, we must have a memory kind.
    if (StringAttr memory_kind = ParseMemoryKind(parser)) {
      return MeshTensorType::get(parser.getContext(), mesh_name_attr.getValue(),
                                 ranked_tensor_type, sharding, memory_kind);
    }
    return {};
  }

  if (!parser.parseGreater()) {
    return MeshTensorType::get(parser.getContext(), mesh_name_attr.getValue(),
                               ranked_tensor_type, sharding);
  }

  return {};
}

void MeshTensorType::print(AsmPrinter& printer) const {
  printer << "<\"" << getMeshName() << "\", ";
  printer.printType(getRankedTensorType());
  if (getSharding()) {
    printer << ", sharding=";
    printer.printStrippedAttrOrType(getSharding());
  }
  if (getMemoryKind()) {
    printer << ", memory_kind=" << getMemoryKind();
  }
  printer << ">";
}

LogicalResult MeshTensorType::verifyForTopology(Operation* op) {
  FailureOr<sdy::MeshAttr> mesh_attr = GetMeshAttr(op, getMeshName());
  return succeeded(mesh_attr) ? verifyForMesh(*mesh_attr, op) : failure();
}

LogicalResult MeshTensorType::verifyForMesh(sdy::MeshAttr mesh_attr,
                                            Operation* op) {
  sdy::TensorShardingAttr sharding = getSharding();
  if (sharding) {
    if (sharding
            .verifyForType(
                getRankedTensorType(), mesh_attr,
                /*emitError=*/
                [op](StringRef msg) { return op->emitOpError(msg); },
                /*checkDivisibility=*/true)
            .failed()) {
      return failure();
    }
  }
  return success();
}

MeshTensorType MeshTensorType::getFullyReplicated(MLIRContext* ctx,
                                                  StringRef mesh_name,
                                                  sdy::MeshAttr mesh,
                                                  RankedTensorType local_type) {
  return MeshTensorType::get(ctx, mesh_name, local_type);
}

MeshTensorType MeshTensorType::replaceSharding(
    sdy::TensorShardingAttr sharding) {
  return MeshTensorType::get(getContext(), getMeshName(), getRankedTensorType(),
                             sharding, getMemoryKind());
}

// Gets the local tensor type of the MeshTensorType wrt the given mesh and
// sharding. Assumes that the sharding is valid wrt the mesh and tensor
// type.
RankedTensorType MeshTensorType::getLocalTensorType(sdy::MeshAttr sdy_mesh) {
  sdy::TensorShardingAttr sharding = getSharding();
  if (!sharding) {
    return getGlobalTensorType();
  }
  return sharding.getLocalTensorType(getGlobalTensorType(), sdy_mesh);
}

RankedTensorType MeshTensorType::getLocalTensorType(Operation* op) {
  auto func_op = sdy::getEnclosingOfType<FuncOp>(op);
  if (HasHomogeneousTopology(func_op)) {
    // TODO(b/439770762): Remove this once we have correct global meshes.
    return MeshTensorType::getLocalTensorType(
        GetTopologyMeshes(func_op).front().getMesh());
  }
  sdy::TensorShardingAttr sharding = getSharding();
  if (!sharding) {
    return getGlobalTensorType();
  }
  // TODO(petebu): Consider looking up the mesh in the global mesh registry.
  return MeshTensorType::getLocalTensorType(
      GetMeshOrFail(op, sharding.getMeshName()));
}

// Functions for the ShapedTypeInterface.
Type MeshTensorType::getElementType() const {
  return getRankedTensorType().getElementType();
}

bool MeshTensorType::hasRank() const { return true; }

ArrayRef<int64_t> MeshTensorType::getShape() const {
  return getRankedTensorType().getShape();
}

// If a new shape is provided, the user intended to change the shape and we
// create a new type without sharding. Otherwise, we create a new type with the
// current shape and sharding.
ShapedType MeshTensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                     Type elementType) const {
  return MeshTensorType::get(
      getContext(), getMeshName(),
      RankedTensorType::get(shape.value_or(getShape()), elementType),
      shape.has_value() ? nullptr : getSharding(), getMemoryKind());
}

bool MeshTensorType::isOnHost() {
  return getMemoryKind() && getMemoryKind().getValue() == kMemoryKindPinnedHost;
}

//===----------------------------------------------------------------------===//
// UserOriginAttr
//===----------------------------------------------------------------------===//

namespace {

void printOptionalTransposeCount(llvm::raw_ostream& os,
                                 int64_t transpose_count) {
  if (transpose_count > 0) {
    os << "(" << transpose_count << ")";
  }
}

}  // namespace

void printOptionalTransposeCount(AsmPrinter& printer, int64_t transpose_count) {
  printOptionalTransposeCount(printer.getStream(), transpose_count);
}

void UserOriginAttr::printShort(llvm::raw_ostream& os) {
  os << "\"" << getUserName().strref() << "\"";
  printOptionalTransposeCount(os, getTransposeCount());
}

void UserOriginAttr::printShort(AsmPrinter& printer) {
  printer.printAttribute(getUserName());
  printOptionalTransposeCount(printer, getTransposeCount());
}

ParseResult parseOptionalTransposeCount(AsmParser& parser,
                                        int64_t& transpose_count) {
  transpose_count = 0;
  if (!parser.parseOptionalLParen()) {
    if (!parser.parseInteger(transpose_count) && !parser.parseRParen()) {
      return success();
    }
    return parser.emitError(parser.getCurrentLocation())
           << "could not parse transpose count value.";
  }
  return success();
}

Attribute UserOriginAttr::parseShort(AsmParser& parser) {
  std::string name;
  int64_t transpose_count = 0;
  if (parser.parseString(&name) ||
      parseOptionalTransposeCount(parser, transpose_count)) {
    parser.emitError(parser.getCurrentLocation())
        << "could not parse user origin.";
    return UserOriginAttr();
  }
  return UserOriginAttr::get(parser.getContext(),
                             StringAttr::get(parser.getContext(), name),
                             transpose_count);
}

namespace {

ParseResult parseFunctionalTypeAndResolveOperands(
    OpAsmParser& parser, OperationState& result,
    llvm::ArrayRef<OpAsmParser::UnresolvedOperand> operands) {
  SmallVector<Type> operand_types;
  SmallVector<Type> result_types;
  if (parser.parseColon() || parser.parseLParen() ||
      (parser.parseOptionalRParen() &&
       (parser.parseTypeList(operand_types) || parser.parseRParen())) ||
      parser.parseArrowTypeList(result_types)) {
    return failure();
  }

  SmallVector<Value> operand_values;
  if (parser.resolveOperands(operands, operand_types,
                             parser.getCurrentLocation(), operand_values)) {
    return failure();
  }

  result.addOperands(operand_values);
  result.addTypes(result_types);

  return success();
}

// Parses an op with a single region with single block and
// verifies the terminator exists. It adds the parsed attrs, operands and result
// types to the parsing result. This is an example of what it parses:
//
// (%opArg1, ..., %opArgN) { optional-attr-dict } (%blockArg1, ..., %blockArgM)
// {
//   // ops in the block
// } : (inputType1, ..., inputTypeN) -> (resultType1, ..., resultTypeK)
ParseResult parseSingleBlockRegionOp(OpAsmParser& parser,
                                     OperationState& result,
                                     NamedAttrList attrs) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren)) {
    return failure();
  }

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(attrs)) {
    return failure();
  }

  if (sdy::parseSingleBlockRegionNoBlockId(parser, *result.addRegion())) {
    return failure();
  }

  if (parseFunctionalTypeAndResolveOperands(parser, result, operands)) {
    return failure();
  }

  result.addAttributes(attrs.getAttrs());

  return success();
}

// Prints a single region op with a single block without the block id, for
// example:
//
// (%opArg1, ..., %opArgN) { optional-attr-dict } (%blockArg1, ..., %blockArgM)
// {
//   // ops in the block
// } : (inputType1, ..., inputTypeN) -> (resultType1, ..., resultTypeK)
//
// The `excluded_attr_names` are omitted from the optional-attr-dict and are
// expected to be handled elsewhere.
void printSingleBlockRegionOp(OpAsmPrinter& p, Operation* op,
                              ArrayRef<StringRef> excluded_attr_names = {}) {
  p << " (";
  p.printOperands(op->getOperands());
  p << ")";

  // Print optional attributes.
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/excluded_attr_names);

  p << " ";
  sdy::printSingleBlockRegionNoBlockId(p, op, op->getRegion(0));

  p << " : ";
  p.printFunctionalType(op);
}

LogicalResult AllInnerAndOuterTypesMatchInNamedComputation(
    NamedComputationOp op, TypeRange inner_types, TypeRange outer_types,
    StringRef inner_name, StringRef outer_name) {
  if (inner_types.size() != outer_types.size()) {
    return op.emitError("number of ")
           << inner_name << "s must match the number of " << outer_name
           << "s respectively: " << inner_types.size()
           << " != " << outer_types.size();
  }

  for (auto [inner_type, outer_type] : llvm::zip(inner_types, outer_types)) {
    if (inner_type != outer_type) {
      return op.emitError("expected the type of the ")
             << inner_name << " to match the type of " << outer_name << ": "
             << inner_type << ", got: " << outer_type;
    }
  }

  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// NamedComputationOp
//===----------------------------------------------------------------------===//

LogicalResult NamedComputationOp::verify() {
  if (failed(AllInnerAndOuterTypesMatchInNamedComputation(
          *this, getBody()->getArgumentTypes(), getOperandTypes(),
          "block argument", "operand")) ||
      failed(AllInnerAndOuterTypesMatchInNamedComputation(
          *this, getBody()->getTerminator()->getOperandTypes(),
          getResultTypes(), "returned value", "result"))) {
    return failure();
  }

  return success();
}

// mpmd.named_computation<"name"> (%op1,..,%opN)
//   (%arg1, ..., %argN) {
//  ...
// } : type
void NamedComputationOp::print(OpAsmPrinter& p) {
  p << "<";
  getOrigin().printShort(p);
  p << ">";

  printSingleBlockRegionOp(p, *this,
                           /*excluded_attr_names=*/{"name", "origin"});
}

ParseResult NamedComputationOp::parse(OpAsmParser& parser,
                                      OperationState& result) {
  NamedAttrList attrs;

  if (parser.parseLess()) {
    return failure();
  }
  auto origin = UserOriginAttr::parseShort(parser);
  if (!origin) {
    return failure();
  }
  if (parser.parseGreater()) {
    return failure();
  }
  attrs.set("origin", origin);
  if (parseSingleBlockRegionOp(parser, result, attrs)) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FragmentOp
//===----------------------------------------------------------------------===//

namespace {

// Checks that the `inner_type` and the `outer_type` of a FragmentOp or
// FragmentCallOp match with each other. When we are verifying the inputs of the
// fragment, `inner_type` is the type of a BlockArgument and `outer_type` is the
// type of a fragment's operator. When we are verifying the outputs of the
// fragment, `inner_type` is the type of an operand of the fragment's return and
// `outer_type` is the type of a fragment's result.
//
// The `outer_type` is expected to be a a valid MeshTensorType w.r.t. the mesh,
// such that its global tensor type must be identical to `inner_type`, which is
// expected to be a RankedTensorType.
template <class FragmentOpTy>
LogicalResult InnerAndOuterTypesMatchInFragment(FragmentOpTy op,
                                                Type inner_type,
                                                Type outer_type,
                                                StringRef inner_name,
                                                StringRef outer_name) {
  StringRef mesh_name = op.getMeshName();
  FailureOr<sdy::MeshAttr> mesh_attr = GetMeshAttr(op, mesh_name);
  if (failed(mesh_attr)) {
    return failure();
  }
  auto mesh_type = cast<MeshTensorType>(outer_type);
  if (mesh_type.getMeshName() != mesh_name) {
    return op.emitError("expected the mesh name of the ")
           << outer_name
           << " to match that of the fragment op: " << mesh_type.getMeshName()
           << " != " << mesh_name;
  }
  if (mesh_type.verifyForMesh(*mesh_attr, op).failed()) {
    return failure();
  }

  if (inner_type != mesh_type.getGlobalTensorType()) {
    return op.emitError("expected the type of the ")
           << inner_name << " to be global tensor type " << outer_name << ": "
           << mesh_type.getGlobalTensorType() << ", got: " << inner_type;
  }

  return success();
}

template <class FragmentOpTy>
LogicalResult AllInnerAndOuterTypesMatchInFragment(FragmentOpTy op,
                                                   TypeRange inner_types,
                                                   TypeRange outer_types,
                                                   StringRef inner_name,
                                                   StringRef outer_name) {
  if (inner_types.size() != outer_types.size()) {
    return op.emitError("number of ")
           << inner_name << "s must match the number of " << outer_name
           << "s respectively: " << inner_types.size()
           << " != " << outer_types.size();
  }

  for (auto [inner_type, outer_type] : llvm::zip(inner_types, outer_types)) {
    if (failed(InnerAndOuterTypesMatchInFragment(op, inner_type, outer_type,
                                                 inner_name, outer_name))) {
      return failure();
    }
  }

  return success();
}

}  // namespace

LogicalResult FragmentOp::verify() {
  ReturnOp return_op = cast<ReturnOp>(getBody()->getTerminator());
  return success(AllInnerAndOuterTypesMatchInFragment(
                     *this, getBody()->getArgumentTypes(), getOperandTypes(),
                     "block argument", "operand")
                     .succeeded() &&
                 AllInnerAndOuterTypesMatchInFragment(
                     *this, return_op.getOperandTypes(), getResultTypes(),
                     "returned value", "result")
                     .succeeded());
}

template <typename OpT>
void printStage(OpT op, llvm::raw_ostream& os) {
  if (op.getStageIdAttr()) {
    os << ", stage=";
    os << op.getStageIdAttr().getInt();
  }
}

template <typename OpT>
void printShardings(OpT op, OpAsmPrinter& p,
                    std::optional<sdy::TensorShardingPerValueAttr> shardings,
                    StringRef sharding_attr_name) {
  if (!shardings) {
    return;
  }
  p << ", " << sharding_attr_name << "=";
  sdy::printStrippedTensorShardingPerValueAttr(p, op, *shardings);
}

template <typename OpT>
void printMeshAndOrigin(OpT op, llvm::raw_ostream& os) {
  os << "mesh=\"" << op.getMeshName() << "\"";
  os << ", origin=[";
  llvm::interleaveComma(op.getOriginAttr(), os, [&](Attribute attr) {
    cast<UserOriginAttr>(attr).printShort(os);
  });
  os << "]";
}

void FragmentOp::printFragmentMetadata(llvm::raw_ostream& os) {
  printMeshAndOrigin<FragmentOp>(*this, os);
  printStage<FragmentOp>(*this, os);
}

// mpmd.fragment<mesh="mesh_name", origin=[... origin ...]>
//  (%op1,..,%opN) dict-attrs
// (%arg1, ..., %argN) {
void FragmentOp::print(OpAsmPrinter& p) {
  p << "<";
  printFragmentMetadata(p.getStream());

  printShardings(*this, p, getInShardings(), "in_shardings");
  printShardings(*this, p, getOutShardings(), "out_shardings");
  p << ">";
  printSingleBlockRegionOp(
      p, *this,
      /*excluded_attr_names=*/
      {"mesh_name", "origin", "stage_id", "in_shardings", "out_shardings"});
}

namespace {

ParseResult parseMeshAndOrigin(OpAsmParser& parser, NamedAttrList& attrs) {
  StringAttr mesh_name_attr;
  std::vector<Attribute> user_origins;

  if (parser.parseKeyword("mesh") || parser.parseEqual() ||
      parser.parseAttribute(mesh_name_attr, "mesh_name", attrs))
    return failure();

  auto parse_user_origin_fn = [&]() -> ParseResult {
    Attribute attr = UserOriginAttr::parseShort(parser);
    if (!attr) return failure();
    user_origins.push_back(attr);
    return success();
  };

  if (parser.parseComma() || parser.parseKeyword("origin") ||
      parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     parse_user_origin_fn)) {
    return failure();
  }
  attrs.set("origin", ArrayAttr::get(parser.getContext(), user_origins));
  return success();
}

ParseResult parseOptionalStage(OpAsmParser& parser, NamedAttrList& attrs,
                               bool& parsed_comma_only) {
  IntegerAttr stage_attr = IntegerAttr();
  if (parser.parseOptionalComma()) {
    return success();
  }
  if (parser.parseOptionalKeyword("stage")) {
    parsed_comma_only = true;
    return success();
  }
  // The "stage" keyword is present, meaning the stage attribute must be
  // defined.
  if (parser.parseEqual() ||
      parser.parseAttribute(stage_attr, "stage_id", attrs)) {
    return failure();
  }
  return success();
}

// Assumes the sharding attribute and keyword are both sharding_attr_name.
ParseResult parseOptionalShardings(OpAsmParser& parser, NamedAttrList& attrs,
                                   bool& parsed_comma_only,
                                   StringRef sharding_attr_name) {
  // Only parse the optional comma if we haven't parsed it.
  if (!parsed_comma_only && parser.parseOptionalComma()) {
    return success();
  }

  if (parser.parseOptionalKeyword(sharding_attr_name)) {
    parsed_comma_only = true;
    return success();
  }
  parsed_comma_only = false;

  // This means that the sharding_attr_name attribute must be defined.
  if (parser.parseEqual()) {
    return failure();
  }
  sdy::TensorShardingPerValueAttr shardingPerValue;
  if (sdy::parseStrippedTensorShardingPerValueAttr(parser, shardingPerValue)) {
    return failure();
  } else {
    attrs.set(sharding_attr_name, shardingPerValue);
  }
  return success();
}

}  // namespace

ParseResult FragmentOp::parse(OpAsmParser& parser, OperationState& result) {
  NamedAttrList attrs;
  bool parsed_comma_only = false;
  // TODO: b/360076171 - Consider parsing a dictionary of optional attributes.
  if (parser.parseLess() || parseMeshAndOrigin(parser, attrs) ||
      parseOptionalStage(parser, attrs, parsed_comma_only) ||
      parseOptionalShardings(parser, attrs, parsed_comma_only,
                             "in_shardings") ||
      parseOptionalShardings(parser, attrs, parsed_comma_only,
                             "out_shardings") ||
      parser.parseGreater() ||
      parseSingleBlockRegionOp(parser, result, attrs)) {
    return failure();
  }
  return success();
}

// Overrides of `Sdy_ShardableDataFlowOpInterface` functions.
SmallVector<sdy::TensorShardingAttr>
FragmentOp::getBlockArgumentEdgeOwnerShardings() {
  if (std::optional<sdy::TensorShardingPerValueAttr> in_shardings =
          getInShardings()) {
    return llvm::to_vector(in_shardings->getShardings());
  }
  return {};
}

SmallVector<sdy::TensorShardingAttr>
FragmentOp::getOpResultEdgeOwnerShardings() {
  if (std::optional<sdy::TensorShardingPerValueAttr> out_shardings =
          getOutShardings()) {
    return llvm::to_vector(out_shardings->getShardings());
  }
  return {};
}

void FragmentOp::setBlockArgumentEdgeOwnerShardings(
    ArrayRef<sdy::TensorShardingAttr> shardings) {
  setInShardingsAttr(
      sdy::TensorShardingPerValueAttr::get(getContext(), shardings));
}

void FragmentOp::setOpResultEdgeOwnerShardings(
    ArrayRef<sdy::TensorShardingAttr> shardings) {
  setOutShardingsAttr(
      sdy::TensorShardingPerValueAttr::get(getContext(), shardings));
}

ArrayRef<BlockArgument> FragmentOp::getBlockArgumentEdgeOwners() {
  return getBody()->getArguments();
}

ResultRange FragmentOp::getOpResultEdgeOwners() { return getResults(); }

// Sets the sharding of a specific result of the fragment.
void FragmentOp::setUserSpecifiedResultSharding(
    unsigned result_index, sdy::TensorShardingAttr new_sharding) {
  if (!new_sharding) {
    return;
  }
  std::optional<sdy::TensorShardingPerValueAttr> current_result_sharding =
      getOutShardings();

  // If none of the results have a sharding, create a new one. Otherwise,
  // replace the sharding of the result at the specified index.
  if (!current_result_sharding) {
    setOutShardingsAttr(
        sdy::TensorShardingPerValueAttr::getOpenWithShardingAtIndex(
            getContext(), getResultTypes(), result_index, new_sharding));
  } else {
    setOutShardingsAttr(current_result_sharding->replaceValueSharding(
        result_index, new_sharding));
  }
}

void FragmentOp::setInputSharding(unsigned input_index,
                                  sdy::TensorShardingAttr sharding) {
  if (!sharding) {
    return;
  }
  std::optional<sdy::TensorShardingPerValueAttr> current_shardings =
      getInShardings();

  // If none of the results have a sharding, create a new one. Otherwise,
  // replace the sharding of the result at the specified index.
  if (!current_shardings) {
    setInShardingsAttr(
        sdy::TensorShardingPerValueAttr::getOpenWithShardingAtIndex(
            getContext(), getResultTypes(), input_index, sharding));
  } else {
    setInShardingsAttr(
        current_shardings->replaceValueSharding(input_index, sharding));
  }
}

// Gets the sources given a target value which need not be an edge owner. Note
// that the return values is a vector, for fragments there can only be one
// value but sdy's interface expects a vector. For example, given the
// following fragment,
// ```
// %r = fragment (%arg0) (%operand0) {
//   %a = some_op
//   return %a
// }
// ```
// If the target is a block argument (e.g., `%operand0`), return `%arg0`.
// If the target is a result (e.g., `%r`), return `%a`.
SmallVector<OpOperand*> FragmentOp::getEdgeSources(Value owner) {
  SDY_CHECK_EQ(sdy::getOwningOp(owner), getOperation());
  if (auto op_result = dyn_cast<OpResult>(owner)) {
    return {
        &getBody()->getTerminator()->getOpOperand(op_result.getResultNumber())};
  }
  return {
      &getOperation()->getOpOperand(cast<BlockArgument>(owner).getArgNumber())};
}

// Returns the edge owner value given a `target`. For fragments, there is only
// one target per data flow edge which is also the edge owner.
Value FragmentOp::getEdgeOwnerFromTarget(Value target) {
  // Check the target is owned by the fragment.
  SDY_CHECK_EQ(sdy::getOwningOp(target), getOperation());
  return target;
}

// Returns the edge owner given a `source`.
// Given the following fragment
// ```
// %r = fragment (%arg0) (%operand0) {
//  %a = some_op
//  return %a
// }
// ```
//
// If the `source` is an operand of a return op, return the corresponding
// result. Otherwise it should be an operand of the fragment, return the block
// argument with the same index.
Value FragmentOp::getEdgeOwnerFromSource(OpOperand& source) {
  Operation* source_owner = source.getOwner();
  if (source_owner->hasTrait<OpTrait::IsTerminator>()) {
    SDY_CHECK_EQ(source_owner->getParentOp(), getOperation());
    return getResult(source.getOperandNumber());
  } else {
    SDY_CHECK_EQ(source_owner, getOperation());
    return getBody()->getArgument(source.getOperandNumber());
  }
}

bool FragmentOp::shouldKeepEdgeOwnerShardingsDivisible() { return true; }

namespace {
FragmentOp CreateMeshFragmentWithBody(
    Location loc, ArrayRef<Attribute> user_origin, StringRef mesh_name,
    ValueRange tensors, TypeRange result_types, OpBuilder& builder,
    FragmentOpBodyPopulator body_populator,
    std::function<Type(Value, sdy::MeshAttr)> get_arg_type) {
  auto origin_attr = ArrayAttr::get(builder.getContext(), user_origin);
  // Only user defined fragments can be assigned to a stage and any fragment
  // created by the compiler is considered to be an inferred fragment.
  // Therefore, the created fragment isn't assigned to a stage.
  FragmentOp fragment_op = FragmentOp::create(builder, loc, result_types,
                                              tensors, origin_attr, mesh_name,
                                              /*stage_id=*/IntegerAttr());
  Block& fragment_block = fragment_op.getRegion().emplaceBlock();
  sdy::MeshAttr mesh_attr = GetMeshOrFail(fragment_op, mesh_name);

  for (Value operand : tensors) {
    fragment_block.addArgument(get_arg_type(operand, mesh_attr),
                               operand.getLoc());
  }
  ArrayRef<Value> arguments(fragment_block.args_begin(),
                            fragment_block.args_end());

  OpBuilder block_builder = OpBuilder::atBlockBegin(&fragment_block);
  ReturnOp::create(block_builder, loc,
                   body_populator(arguments, block_builder));
  return fragment_op;
}
}  // namespace

FragmentOp FragmentOp::createMeshFragmentWithGlobalBody(
    Location loc, ArrayRef<Attribute> user_origin, StringRef mesh_name,
    ValueRange tensors, TypeRange result_types, OpBuilder& builder,
    FragmentOpBodyPopulator body_populator) {
  return CreateMeshFragmentWithBody(loc, user_origin, mesh_name, tensors,
                                    result_types, builder, body_populator,
                                    GetGlobalTensorTypeFromMeshType);
}

//===----------------------------------------------------------------------===//
// FragmentCallOp
//===----------------------------------------------------------------------===//

void FragmentCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr()) {
    auto symRef = callee.get<SymbolRefAttr>();
    return setCalleeAttr(cast<FlatSymbolRefAttr>(symRef));
  }
  // Indirect call, callee Value is the first operand.
  return setOperand(0, callee.get<Value>());
}

LogicalResult FragmentCallOp::verifySymbolUses(
    SymbolTableCollection& symbolTable) {
  // Check that the callee references a valid function.
  auto func_op =
      symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, getCalleeAttr());
  if (!func_op) {
    return emitError("'") << getCallee()
                          << "' does not reference a valid function";
  }

  FunctionType func_type = func_op.getFunctionType();
  if (failed(AllInnerAndOuterTypesMatchInFragment(
          *this, func_type.getInputs(), getOperandTypes(), "block argument",
          "operand")) ||
      failed(AllInnerAndOuterTypesMatchInFragment(
          *this, func_type.getResults(), getResultTypes(), "returned value",
          "result"))) {
    return failure();
  }

  // We already verified the call mesh is valid above when verifying types.
  sdy::MeshAttr call_mesh = GetMeshOrFail(*this, getMeshName());
  FailureOr<sdy::MeshAttr> func_mesh = GetMeshAttr(func_op);
  if (failed(func_mesh)) {
    return failure();
  } else if (call_mesh != *func_mesh) {
    return emitError(
               "Expected mesh of fragment call and callee function to "
               "match: ")
           << call_mesh << " vs " << *func_mesh;
  } else {
    return success();
  }
}

// mpmd.fragment_call<mesh="mesh_name", origin=[... origin ...]>
//   @callee(%op1,..,%opN) dict-attrs : functional-type
void FragmentCallOp::print(OpAsmPrinter& p) {
  p << "<";
  printMeshAndOrigin(*this, p.getStream());
  p << ">";
  p << " ";
  p.printSymbolName(getCalleeAttr().getValue());
  p << "(";
  p.printOperands(getOperands());
  p << ")";
  // Print optional attributes.
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*excluded_attr_names=*/{"mesh_name", "origin", "callee"});
  p << " : ";
  p.printFunctionalType(getOperation());
}

ParseResult FragmentCallOp::parse(OpAsmParser& parser, OperationState& result) {
  NamedAttrList attrs;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SymbolRefAttr callee;
  if (parser.parseLess() || parseMeshAndOrigin(parser, attrs) ||
      parser.parseGreater() || parser.parseAttribute(callee, "callee", attrs) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(attrs) ||
      parseFunctionalTypeAndResolveOperands(parser, result, operands)) {
    return failure();
  }
  result.addAttributes(attrs.getAttrs());
  return success();
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

bool TransferOp::isIntraMesh() {
  return getTensor().getType().getMeshName() ==
             getResult().getType().getMeshName() &&
         getTensor().getType().getMemoryKind() ==
             getResult().getType().getMemoryKind();
}

namespace {

StringAttr FindMemoryKindInAttributes(Value value, FuncOp func) {
  if (auto block_arg = dyn_cast<BlockArgument>(value)) {
    return func.getArgAttrOfType<StringAttr>(block_arg.getArgNumber(),
                                             kMemoryKindAttr);
  }

  if (auto transfer_producer = dyn_cast<TransferOp>(value.getDefiningOp())) {
    return transfer_producer->getAttrOfType<StringAttr>(kMemoryKindAttr);
  }

  // Operations with multiple results.
  if (isa<FragmentOp, CallOp, FragmentCallOp, ForOp>(value.getDefiningOp())) {
    return dyn_cast_if_present<StringAttr>(
        GetResAttr(value.getDefiningOp(),
                   cast<OpResult>(value).getResultNumber(), kMemoryKindAttr));
  }
  return nullptr;
}

}  // namespace

LogicalResult TransferOp::verify() {
  auto mesh_type_in = cast<MeshTensorType>(getTensor().getType());
  auto mesh_type_out = cast<MeshTensorType>(getResult().getType());

  if (mesh_type_in.verifyForTopology(getOperation()).failed() ||
      mesh_type_out.verifyForTopology(getOperation()).failed()) {
    return failure();
  }
  if (mesh_type_in.getRankedTensorType() !=
      mesh_type_out.getRankedTensorType()) {
    return emitError("cannot perform transfer between given types ")
           << mesh_type_in << " and " << mesh_type_out
           << ": global tensor types are different";
  }

  if (StringAttr in_memory_kind = mesh_type_in.getMemoryKind()) {
    if (in_memory_kind.getValue() != kMemoryKindPinnedHost &&
        in_memory_kind.getValue() != kMemoryKindDevice) {
      return emitError("memory kind must be either '")
             << kMemoryKindPinnedHost << "' or '" << kMemoryKindDevice
             << "'. Found '" << in_memory_kind.getValue() << "'.";
    }
  }

  if (StringAttr out_memory_kind = mesh_type_out.getMemoryKind()) {
    if (out_memory_kind.getValue() != kMemoryKindPinnedHost &&
        out_memory_kind.getValue() != kMemoryKindDevice) {
      return emitError("memory kind must be either '")
             << kMemoryKindPinnedHost << "' or '" << kMemoryKindDevice
             << "'. Found '" << out_memory_kind.getValue() << "'.";
    }
  }

  // TODO: b/399865449 - We should not rely on attributes for host offloading.
  // Instead, we should use the memory kind in the type.
  StringAttr in_memory_kind = FindMemoryKindInAttributes(
      getTensor(), getOperation()->getParentOfType<FuncOp>());
  if (in_memory_kind && in_memory_kind.getValue() == kMemoryKindPinnedHost) {
    return emitError(
        "Transfers from host with attributes are not supported. Memory kinds "
        "must be expressed in the type.");
  }
  StringAttr out_memory_kind = FindMemoryKindInAttributes(
      getResult(), getOperation()->getParentOfType<FuncOp>());
  if (out_memory_kind && out_memory_kind.getValue() == kMemoryKindPinnedHost) {
    return emitError(
        "Transfers to host with attributes are not supported. Memory kinds "
        "must expressed be in the type.");
  }

  return success();
}

sdy::OpShardingRuleAttr TransferOp::getShardingRule() {
  return sdy::OpShardingRuleBuilder::buildPointwise(*this);
}

bool TransferOp::shouldKeepOutputShardingsDivisible() { return true; }

namespace {

// Verifies all of the following:
// 1. The given `mesh_type` is valid.
// 2. `mesh_type` is fully replicated.
// 3. The given `local_type` matches the distributed type of `mesh_type`.
LogicalResult VerifyAssignmentOpTypes(Operation* op,
                                      RankedTensorType local_type,
                                      MeshTensorType mesh_type) {
  if (mesh_type.verifyForTopology(op).failed()) {
    return failure();
  }

  // We expect the distributed types to be fully replicated because the
  // assign/unassign ops are only present before import, when the program is
  // still unpartitioned.
  if (!mesh_type.isFullyReplicated()) {
    return op->emitError() << "MeshTensorType should be fully replicated: "
                           << mesh_type;
  }

  if (mesh_type.getRankedTensorType() != local_type) {
    return op->emitError("mismatch between the given local type ")
           << local_type << " and mesh_type ranked tensor type: "
           << mesh_type.getRankedTensorType();
  }

  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

LogicalResult AssignOp::verify() {
  auto local_type_in = cast<RankedTensorType>(getTensor().getType());
  auto mesh_type_out = cast<MeshTensorType>(getResult().getType());

  return VerifyAssignmentOpTypes(getOperation(), local_type_in, mesh_type_out);
}

//===----------------------------------------------------------------------===//
// UnassignOp
//===----------------------------------------------------------------------===//

LogicalResult UnassignOp::verify() {
  auto mesh_type_in = cast<MeshTensorType>(getTensor().getType());
  auto local_type_out = cast<RankedTensorType>(getResult().getType());

  return VerifyAssignmentOpTypes(getOperation(), local_type_out, mesh_type_in);
}

LogicalResult UnassignOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    llvm::SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  UnassignOp::Adaptor adaptor(operands, attributes, properties, regions);
  inferredReturnShapes.emplace_back(
      cast<ShapedType>(cast<MeshTensorType>(adaptor.getTensor().getType())
                           .getRankedTensorType()));
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr()) {
    auto symRef = callee.get<SymbolRefAttr>();
    setCalleeAttr(cast<FlatSymbolRefAttr>(symRef));
  }
  // Indirect call, callee Value is the first operand.
  setOperand(0, callee.get<Value>());
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  // Check that the callee references a valid function.
  auto func_op =
      symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, getCalleeAttr());
  if (!func_op) {
    return emitError("No function was found for function ref '")
           << getCallee() << "'";
  }

  FunctionType func_type = func_op.getFunctionType();
  TypeRange operand_types = getOperandTypes();
  for (auto [function_input_type, operand_type] :
       llvm::zip(func_type.getInputs(), operand_types)) {
    if (function_input_type != operand_type) {
      return emitError("Type mismatch. Expected call operand to have type ")
             << function_input_type << " but got " << operand_type;
    }
  }

  TypeRange result_types = getResultTypes();
  for (auto [function_result_type, call_result_type] :
       llvm::zip(func_type.getResults(), result_types)) {
    if (function_result_type != call_result_type) {
      return emitError("Type mismatch. Expected call result to have type ")
             << function_result_type << " but got " << call_result_type;
    }
  }

  if (func_op->hasAttr(kTopologyAttr) && !func_op.isPrivate()) {
    return emitError(
        "MPMD CallOp callee with topology must also have visibility set to "
        "private, as public functions with topologies are assumed to be "
        "entry "
        "point functions.");
  }

  return success();
}

LogicalResult CallOp::verify() {
  if (auto count = (*this)->getAttrOfType<IntegerAttr>(kCallCounterAttrName)) {
    if (!count.getType().isUnsignedInteger(32)) {
      return emitError() << "call_counter must be an uint32, got "
                         << count.getType();
    }
  }

  Operation* parent_op = (*this)->getParentOp();
  FuncOp parent_func = (*this)->getParentOfType<FuncOp>();

  if (!isa<FuncOp, ForOp>(parent_op)) {
    return emitError() << "Mpmd CallOp on \"" << (*this).getCallee()
                       << "\" can only be used in a function or for_op block "
                          "but was called from inside op "
                       << parent_op->getName().getStringRef();
  }

  if (IsMpmdFunction(parent_func) && !IsEntryPointFunction(parent_func)) {
    return emitError()
           << "Mpmd CallOp on \"" << (*this).getCallee()
           << "\" in an Mpmd function can only be used directly by "
              "the entrypoint function, i.e. the main function, but "
              "was called from \""
           << parent_func.getSymName() << "\".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

ParseResult ForOp::parse(OpAsmParser& parser, OperationState& result) {
  auto loc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren)) {
    return failure();
  }

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs)) {
    return failure();
  }
  result.addAttributes(attrs.getAttrs());

  if (sdy::parseSingleBlockRegionNoBlockId(parser, *result.addRegion())) {
    return failure();
  }

  SmallVector<Type> result_types;
  if (parser.parseColon() || parser.parseTypeList(result_types)) {
    return failure();
  }
  result.addTypes(result_types);

  SmallVector<Value> operand_values;
  if (parser.resolveOperands(operands, result_types, loc, operand_values)) {
    return failure();
  }
  result.addOperands(operand_values);

  return success();
}

void ForOp::print(OpAsmPrinter& p) {
  p << " (";
  p.printOperands(getOperands());
  p << ")";

  p.printOptionalAttrDict((*this)->getAttrs());

  p << " ";
  sdy::printSingleBlockRegionNoBlockId(p, *this, getRegion());

  p << " : ";
  for (const auto& result_type : llvm::enumerate(getResultTypes())) {
    p.printType(result_type.value());
    if (result_type.index() != getNumResults() - 1) {
      p << ", ";
    }
  }
}

LogicalResult ForOp::verify() {
  if (getRegion().getNumArguments() != getNumOperands() + 1) {
    return emitError("wrong number of arguments for region");
  }

  // Below we will perform type checking.
  // Note: no need to explicitly check that operands and ForOp result types
  // match because this is implied by HLO_PairwiseSameOperandAndResultType.
  // Therefore we only check that the operands match the block arguments (except
  // for the last, which is the loop index), and the results match the
  // terminator operand types.
  for (int i = 0; i < getNumOperands(); ++i) {
    if (getOperand(i).getType() != getRegion().getArgument(i).getType()) {
      return emitError("wrong argument type at argument no. ") << i;
    }
  }
  // Permit other types for the index argument so that we can use MeshTensor.
  // TODO(petebu): Break circular dependency between dialects and add check for
  // MeshTensor.
  if (auto index_type = dyn_cast<RankedTensorType>(getIndexArg().getType())) {
    if (!index_type || index_type.getRank() != 0 ||
        !index_type.getElementType().isInteger(32)) {
      return emitError("index must have a 32-bit integer rank-0 tensor type");
    }
  }
  Operation* terminator = getBody()->getTerminator();
  if (terminator->getOperandTypes() != getResultTypes()) {
    return emitError("type mismatch for result of region");
  }

  // Add a strict check that all types come from the same class. We may consider
  // lifting this if we support e.g. tensors and tokens in the future.
  if (getNumOperands() > 0) {
    TypeID type_id = getOperand(0).getType().getTypeID();
    if (!llvm::all_of(getOperandTypes(), [&type_id](Type operand_type) {
          return operand_type.getTypeID() == type_id;
        })) {
      return emitError("type ids in operands/results are not identical");
    }
  }

  if (getIterations() <= 0) {
    return emitError("number of iterations must be greater than zero");
  }

  int factor = getUnrollFactor().value_or(1);
  if (factor == 0 || getIterations() % factor != 0) {
    return emitError("number of iterations ")
           << getIterations() << " isn't divisible by unroll factor " << factor;
  }

  return success();
}

ForOp ForOp::create(Location loc, ValueRange tensors, uint32_t iterations,
                    OpBuilder& builder, ForOpBodyPopulator body_populator,
                    uint32_t unroll_factor) {
  TypeRange result_types = tensors.getTypes();
  auto op = ForOp::create(
      builder, loc, result_types, tensors, iterations,
      unroll_factor == 1 ? nullptr : builder.getUI32IntegerAttr(unroll_factor));

  Block& block = op.getRegion().emplaceBlock();
  for (Value operand : tensors) {
    block.addArgument(operand.getType(), operand.getLoc());
  }
  block.addArgument(
      RankedTensorType::get({}, builder.getIntegerType(32, /*isSigned=*/false)),
      loc);

  ArrayRef<Value> args(block.args_begin(), block.args_end());

  OpBuilder block_builder = OpBuilder::atBlockBegin(&block);
  ReturnOp::create(
      block_builder, loc,
      body_populator(args.drop_back(), /*index=*/args.back(), block_builder));
  return op;
}

//===------------------------------------------------------------------===//
// ShardableDataFlowOpInterface methods
//===------------------------------------------------------------------===//
ResultRange ForOp::getOpResultEdgeOwners() { return getResults(); }

// Returns a list of sources given an edge `owner`.
SmallVector<OpOperand*> ForOp::getEdgeSources(Value owner) {
  auto op_result = cast<OpResult>(owner);
  SDY_CHECK(op_result.getOwner() == getOperation());
  unsigned res_num = op_result.getResultNumber();
  return {&getOperation()->getOpOperand(res_num),
          &getYieldedValuesMutable().value()[res_num]};
}

SmallVector<Value> ForOp::getNonEdgeOwnerTargets(Value owner) {
  auto op_result = cast<OpResult>(owner);
  SDY_CHECK(op_result.getOwner() == getOperation());
  return {getRegionIterArgs()[op_result.getResultNumber()]};
}

// Returns the edge own given a `target`. `target` may not be an edge owner.
Value ForOp::getEdgeOwnerFromTarget(Value target) {
  SDY_CHECK(sdy::getOwningOp(target) == getOperation());
  if (auto op_result = dyn_cast<OpResult>(target)) {
    return op_result;
  }
  SDY_CHECK(isa<BlockArgument>(target));
  return getResult(cast<BlockArgument>(target).getArgNumber());
}

// Returns the edge owner given a `source` of the data flow edge.
Value ForOp::getEdgeOwnerFromSource(OpOperand& source) {
  Operation* source_owner = source.getOwner();
  if (source_owner->hasTrait<OpTrait::IsTerminator>()) {
    SDY_CHECK_EQ(source_owner->getParentOp(), getOperation());
  } else {
    SDY_CHECK_EQ(source_owner, getOperation());
  }
  return getResult(source.getOperandNumber());
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verify() {
  if (getReductionType() == ReductionType::kNone && getNumOperands() != 1) {
    return emitError(
        "ReduceOp must have exactly one operand if the reduction type is none");
  }
  return success();
}

// Replaces a chain of reduces with a single reduce if the outer reduce type
// is none or the reduction types match.
//
// In symbols:
//
//  y = reduce<K> x
//  z = reduce<none> y
//  ~>
//  z = reduce<K> x
// and
//  y = reduce<K> x1, x2
//  z = reduce<K> y
//  ~>
//  z = reduce<K> x1, x2
LogicalResult ReduceOp::canonicalize(ReduceOp op, PatternRewriter& rewriter) {
  Operation* defining_op = op.getOperands().front().getDefiningOp();
  if (!defining_op) {
    return failure();
  }
  ReductionType outer_reduction_type = op.getReductionType();
  if (ReduceOp inner_reduce = llvm::dyn_cast<ReduceOp>(defining_op)) {
    if (inner_reduce.getReductionType() == outer_reduction_type ||
        outer_reduction_type == ReductionType::kNone) {
      rewriter.replaceAllOpUsesWith(op, inner_reduce);
      return success();
    }
  }
  return failure();
}

// Avoids passing values through the fragment just to be used by other
// fragments or transfers. Instead, we want to use those values directly.
// NOTE: this may have benefits from memory usage.
LogicalResult FragmentOp::canonicalize(FragmentOp op,
                                       PatternRewriter& rewriter) {
  Block& block = op.getRegion().front();
  Operation* return_op = block.getTerminator();
  bool result_replaced = false;
  for (BlockArgument arg : block.getArguments()) {
    auto uses = arg.getUses();
    auto it = llvm::find_if(uses, [&return_op](OpOperand& use) {
      return use.getOwner() == return_op;
    });
    if (it != uses.end()) {
      Value fragment_result = op.getResult(it->getOperandNumber());
      if (fragment_result.use_empty()) {
        continue;
      }
      if (op.getOperand(arg.getArgNumber()).getType() !=
          fragment_result.getType()) {
        continue;
      }
      rewriter.replaceAllUsesWith(fragment_result,
                                  op.getOperand(arg.getArgNumber()));
      result_replaced = true;
    }
  }
  return success(result_replaced);
}

}  // namespace mlir::mpmd

using ::mlir::stablehlo::TokenType;  // NOLINT

#include "shardy/dialect/mpmd/ir/dialect.cc.inc"
#include "shardy/dialect/mpmd/ir/enums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "shardy/dialect/mpmd/ir/attrs.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "shardy/dialect/mpmd/ir/types.cc.inc"
#define GET_OP_CLASSES
#include "shardy/dialect/mpmd/ir/ops.cc.inc"

namespace {

#include "shardy/dialect/mpmd/ir/canonicalization.cc.inc"

}  // namespace

namespace mlir::mpmd {

void TransferOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<IdentityTransferPattern, IntraMeshTransferOfTransferPattern>(
      context);
}

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<BroadcastOfBroadcastPattern>(context);
}

}  // namespace mlir::mpmd
