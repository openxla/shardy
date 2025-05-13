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

#include "shardy/dialect/sdy/ir/parsers.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/AssemblyFormat.h"

namespace mlir {
namespace sdy {

Attribute DimensionShardingAttr::parse(AsmParser& parser, Type type) {
  if (parser.parseLBrace()) {
    return DimensionShardingAttr();
  }

  SmallVector<AxisRefAttr> axes;
  bool isClosed = true;

  while (parser.parseOptionalRBrace()) {
    if (!axes.empty() && parser.parseComma()) {
      return DimensionShardingAttr();
    }
    if (!parser.parseOptionalQuestion()) {
      isClosed = false;
      if (parser.parseRBrace()) {
        return DimensionShardingAttr();
      }
      break;
    }
    Attribute axisRef = AxisRefAttr::parse(parser, {});
    if (!axisRef) {
      return DimensionShardingAttr();
    }
    // TODO(tomnatan): remove mlir:: once Attribute::cast is removed.
    axes.push_back(mlir::cast<AxisRefAttr>(axisRef));
  }

  int64_t priority = -1;
  StringRef priorityStr;

  if (!parser.parseOptionalKeyword(&priorityStr)) {
    StringRef priorityNumStr = priorityStr.drop_front();
    if (priorityStr.size() < 2 || priorityStr.front() != 'p' ||
        !llvm::all_of(priorityNumStr, [](char c) { return std::isdigit(c); })) {
      parser.emitError(parser.getCurrentLocation(),
                       "expecting priority in format 'p<number>', got: ")
          << priorityStr;
      return DimensionShardingAttr();
    }
    if (priorityNumStr.starts_with("0") && priorityNumStr.size() > 1) {
      parser.emitError(parser.getCurrentLocation(),
                       "priorities with leading zeros are not allowed, got: ")
          << priorityStr;
      return DimensionShardingAttr();
    }
    if (!llvm::to_integer(priorityNumStr, priority)) {
      parser.emitError(parser.getCurrentLocation(),
                       "expecting integer priority, got: ")
          << priorityStr;
      return DimensionShardingAttr();
    }
  }

  return DimensionShardingAttr::get(
      parser.getContext(), axes, isClosed,
      priority == -1 ? std::nullopt : std::make_optional(priority));
}

ParseResult parseMeshOrRef(AsmParser& parser, Attribute& meshOrRef) {
  if (!parser.parseOptionalKeyword(MeshAttr::getMnemonic())) {
    auto mesh = FieldParser<MeshAttr>::parse(parser);
    if (failed(mesh)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "failed to parse MeshAttr");
    }
    meshOrRef = *mesh;
    return success();
  }

  auto symbolRef = FieldParser<FlatSymbolRefAttr>::parse(parser);
  if (failed(symbolRef)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expecting MeshAttr or FlatSymbolRefAttr, got: ")
           << meshOrRef;
  }
  meshOrRef = *symbolRef;
  return success();
}

namespace {

ParseResult parseEqualsAxisList(AsmParser& parser,
                                SmallVector<AxisRefAttr>& axes) {
  if (parser.parseEqual() || parser.parseLBrace()) {
    return failure();
  }
  auto parsedAxes = FieldParser<SmallVector<AxisRefAttr>>::parse(parser);
  if (failed(parsedAxes)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to parse axis list which is expected to "
                            "be an `ArrayRef<AxisRefAttr>`");
  }
  if (parser.parseRBrace()) {
    return failure();
  }
  axes = std::move(*parsedAxes);
  return success();
}

}  // namespace

ParseResult parseReplicatedAndUnreducedAxes(
    AsmParser& parser, SmallVector<AxisRefAttr>& replicatedAxes,
    SmallVector<AxisRefAttr>& unreducedAxes) {
  while (!parser.parseOptionalComma()) {
    if (replicatedAxes.empty() && !parser.parseOptionalKeyword("replicated")) {
      if (parseEqualsAxisList(parser, replicatedAxes)) {
        return parser.emitError(
            parser.getCurrentLocation(),
            "failed to parse Sdy_TensorSharding parameter 'replicated_axes'");
      }
    } else if (unreducedAxes.empty() &&
               !parser.parseOptionalKeyword("unreduced")) {
      if (parseEqualsAxisList(parser, unreducedAxes)) {
        return parser.emitError(
            parser.getCurrentLocation(),
            "failed to parse Sdy_TensorSharding parameter 'unreduced_axes'");
      }
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "failed to parse Sdy_TensorSharding, expected "
                              "valid named axis list after comma");
    }
  }
  return success();
}

namespace {

// Removes and returns the index from the symbol in factorsStr. For example:
// 'jkl' -> factorsStr set to 'kl` and returns 1
// 'z_2z_1' -> factorsStr set to 'z_1` and returns 27 (2 + 'z')
FailureOr<int64_t> parseFactorSymbolIndex(AsmParser& parser,
                                          StringRef& factorsStr) {
  if (!factorsStr.starts_with("z_")) {
    // Should be a single character like i/j/k/etc (possibly compound!).
    // Get int equivalent.
    char symbol = factorsStr.front();
    if (symbol < 'i' || symbol > 'z') {
      return parser.emitError(parser.getCurrentLocation(),
                              "expecting symbol from 'i' to 'z'. Received: '")
             << std::string(1, symbol) << "'";
    }
    factorsStr = factorsStr.drop_front();
    return symbol - 'i';
  }

  if (factorsStr.size() < 2 || factorsStr.take_front(2) != "z_") {
    return parser.emitError(parser.getCurrentLocation(),
                            "expecting 'z_'. Received: '")
           << factorsStr << "'";
  }
  int64_t symbolIndex = 0;
  size_t nonNumIndex = 2;
  while (nonNumIndex < factorsStr.size() &&
         std::isdigit(factorsStr[nonNumIndex])) {
    nonNumIndex++;
  }
  if (nonNumIndex == 2 || nonNumIndex > factorsStr.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expecting integer after 'z_'. Received: '")
           << factorsStr << "'";
  }
  StringRef numberStr = factorsStr.substr(2, nonNumIndex - 2);
  if (numberStr[0] == '0') {
    return parser.emitError(
               parser.getCurrentLocation(),
               "expecting positive integer without leading zeros. Received: '")
           << numberStr << "'";
  }
  if (!llvm::to_integer(numberStr, symbolIndex)) {
    parser.emitError(parser.getCurrentLocation(),
                     "expecting symbol index <=2^63-1. Received: '")
        << numberStr << "'";
  }
  // Remove z_#
  factorsStr = factorsStr.substr(nonNumIndex);
  // NOTE: if number overflows, there will be undefined behavior. The chances
  // of a tensor of that rank are slim.
  return symbolIndex + kStartAtZ;
}

// Removes and returns the index from the symbol in factorsStr while verifying
// the symbol represents the correct index `expectedFactorIndex`. See
// parseFactorSymbolIndex for what it parses from factorsStr.
// We expect that symbols in a list of factors are in sorted order, increasing
// by 1 each time.
FailureOr<int64_t> parseFactorSymbolIndex(AsmParser& parser,
                                          StringRef& factorsStr,
                                          int64_t expectedFactorIndex) {
  FailureOr<int64_t> factorIndex = parseFactorSymbolIndex(parser, factorsStr);
  if (failed(factorIndex)) {
    return failure();
  }
  // Check the symbol appears in increasing order starting from 0,
  // incrementing by 1 each time. So only `{i=#, j=#, ...}` are accepted.
  // These are all invalid:
  //   - {k=#, i=#, j=#}  // i comes before k
  //   - {m=#, n=#, o=#}  // has to start at i (index 0)
  //   - {i=#, k=#}       // can't skip an index, j (index 1) is missing
  if (*factorIndex != expectedFactorIndex) {
    return parser.emitError(
               parser.getCurrentLocation(),
               "expecting factor indices to be ordered like an iota "
               "([0,1,2,...], "
               "e.g. {i=#, j=#, ...}). Expecting factor index symbol '")
           << factorSymbolString(expectedFactorIndex) << "', received: '"
           << factorSymbolString(*factorIndex) << "'";
  }
  return factorIndex;
}

// Parses dimensions factor mapping indices. In an OpShardingRule, you could
// have `([i, j])->([ij]), this parses each dimension of the operand and result
// tensor, so indices would be set to:
// - `i` -> [0]
// - `j` -> [1]
// - `ij` -> [0, 1]
// for each case.
ParseResult parseSymbolIndices(AsmParser& parser, StringRef factorsStr,
                               SmallVector<int64_t>& indices) {
  while (!factorsStr.empty()) {
    // TODO(bartchr): Add SDY_ASSIGN_OR_RETURN_FAILURE macro for re-returning
    // failures. Or check if there already is one in MLIR.
    FailureOr<int64_t> index = parseFactorSymbolIndex(parser, factorsStr);
    if (failed(index)) {
      return failure();
    }
    indices.push_back(*index);
  }
  return success();
}

}  // namespace

Attribute DimMappingAttr::parse(AsmParser& parser, Type) {
  MLIRContext* context = parser.getContext();

  SmallVector<int64_t> factorIndices;
  StringRef factor;
  if (parser.parseOptionalKeyword(&factor)) {
    // Defer to verification for failure.
    return DimMappingAttr::get(context, {});
  }
  if (parseSymbolIndices(parser, factor, factorIndices).failed()) {
    return DimMappingAttr();
  }

  return DimMappingAttr::get(context, factorIndices);
}

// Parses factor sizes. In a OpShardingRule, you could have `{i=2, j=4}`.
// `i`, which is index 0, is of size 2, while `j` is of size 4. Thus factorSizes
// would be set to [2, 4].
ParseResult parseFactorSizes(AsmParser& parser,
                             SmallVector<int64_t>& factorSizes) {
  int64_t expectedFactorIndex = 0;
  auto parseElementFn = [&]() -> ParseResult {
    StringRef factorSymbol;
    if (parser.parseKeyword(&factorSymbol)) {
      return failure();
    }
    if (failed(parseFactorSymbolIndex(parser, factorSymbol,
                                      expectedFactorIndex++))) {
      return failure();
    }
    if (!factorSymbol.empty()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expecting single factor symbol: ")
             << factorSymbol;
    }

    if (parser.parseEqual()) {
      return failure();
    }

    int factorSize;
    if (parser.parseInteger(factorSize)) {
      return failure();
    }
    factorSizes.push_back(factorSize);
    return success();
  };

  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::OptionalBraces,
                                     parseElementFn)) {
    return failure();
  }
  return success();
}

// Parses factor sizes. In a OpShardingRule, you could have `, type={k, i}`.
// `k` is index 2, while `i` is index 0. Thus factors would be set to [2, 0].
ParseResult parseFactorsWithType(AsmParser& parser,
                                 SmallVector<int64_t>& factors,
                                 StringRef type) {
  auto parseElementFn = [&]() -> ParseResult {
    StringRef factorSymbol;
    if (parser.parseKeyword(&factorSymbol)) {
      return failure();
    }
    FailureOr<int64_t> factorIndex =
        parseFactorSymbolIndex(parser, factorSymbol);
    if (failed(factorIndex)) {
      return failure();
    }
    factors.push_back(*factorIndex);
    return success();
  };

  if (!parser.parseOptionalKeyword(type)) {
    if (parser.parseEqual()) {
      return failure();
    }
    return parser.parseCommaSeparatedList(AsmParser::Delimiter::OptionalBraces,
                                          parseElementFn);
  }
  return success();
}

ParseResult parseIsCustomRule(AsmParser& parser, bool& isCustomRule) {
  isCustomRule = false;
  if (!parser.parseOptionalComma()) {
    if (parser.parseKeyword("custom")) {
      return failure();
    }
    isCustomRule = true;
  }
  return success();
}

ParseResult parseSingleBlockRegionNoBlockId(OpAsmParser& parser,
                                            Region& region) {
  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true,
                               /*allowAttrs=*/true)) {
    return failure();
  }
  if (parser.parseRegion(region, regionArgs)) {
    return failure();
  }

  return success();
}

ParseResult parseStrippedTensorShardingPerValueAttr(
    AsmParser& parser, TensorShardingPerValueAttr& shardingPerValue) {
  SmallVector<TensorShardingAttr> shardings;
  auto parseElementFn = [&shardings, &parser]() -> ParseResult {
    auto element = FieldParser<TensorShardingAttr>::parse(parser);
    if (failed(element)) return failure();
    shardings.push_back(std::move(*element));
    return success();
  };

  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     parseElementFn)) {
    return failure();
  }
  shardingPerValue =
      TensorShardingPerValueAttr::get(parser.getContext(), shardings);
  return success();
}

ParseResult ConstantOp::parse(OpAsmParser& parser, OperationState& result) {
  return hlo::parseConstantOp(parser, result);
}

}  // namespace sdy
}  // namespace mlir
