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

#ifndef SHARDY_DIALECT_SDY_IR_DIALECT_H_
#define SHARDY_DIALECT_SDY_IR_DIALECT_H_

// IWYU pragma: begin_keep

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/verifiers.h"

// IWYU pragma: end_keep

// IWYU pragma: begin_exports

// Dialect main class is defined in ODS, we include it here.
#include "shardy/dialect/sdy/ir/dialect.h.inc"
// ODS-generated enum classes.
#include "shardy/dialect/sdy/ir/enums.h.inc"
// ODS-generated attribute classes.
#define GET_ATTRDEF_CLASSES
#include "shardy/dialect/sdy/ir/attrs.h.inc"

// Below are methods that are the bodies of ODS-generated op-interface classes
// which cannot be inlined due to cyclic dependencies on helper functions.

namespace mlir {
namespace sdy {
namespace details {

// Default implementation of the `getOpResultEdgeOwnerShardings` method of
// `ShardableDataFlowOpInterface`.
SmallVector<TensorShardingAttr> getOpResultEdgeOwnerShardingsImpl(
    Operation* op);

// Default implementation of the `setOpResultEdgeOwnerShardings` method of
// `ShardableDataFlowOpInterface`.
void setOpResultEdgeOwnerShardingsImpl(Operation* op,
                                       ArrayRef<TensorShardingAttr> shardings);

}  // namespace details

struct SdyDialectVersion : public mlir::DialectVersion {
  SdyDialectVersion(int64_t major, int64_t minor, int64_t patch)
      : majorMinorPatch({major, minor, patch}) {}

  int64_t getMajor() const { return majorMinorPatch[0]; }
  int64_t getMinor() const { return majorMinorPatch[1]; }
  int64_t getPatch() const { return majorMinorPatch[2]; }

  bool operator<(const SdyDialectVersion& other) const {
    return this->majorMinorPatch < other.majorMinorPatch;
  }

  // Current version of Shardy dialect.
  static SdyDialectVersion getCurrentVersion() { return {0, 0, 1}; }

  // Parse version in format "123.1235.13"
  // each number is 0-max(int64_t)
  static FailureOr<SdyDialectVersion> fromString(const StringRef& version) {
    if (version == "current") return getCurrentVersion();
    SmallVector<StringRef> parts;
    version.split(parts, /*Separator=*/'.', /*MaxSplit=*/2,
                  /*KeepEmpty=*/true);
    if (parts.size() != 3) return failure();
    int64_t major, minor, patch;
    if (!llvm::to_integer(parts[0], major, 10) ||
        !llvm::to_integer(parts[1], minor, 10) ||
        !llvm::to_integer(parts[2], patch, 10))
      return failure();
    return SdyDialectVersion(
        /*major=*/major, /*minor=*/minor, /*patch=*/patch);
  }

 private:
  // The dialect version read from bytecode.
  std::array<int64_t, 3> majorMinorPatch;
};

}  // namespace sdy

// Allow printing to a stream.
inline raw_ostream& operator<<(raw_ostream& os,
                               sdy::SdyDialectVersion version) {
  os << version.getMajor() << "." << version.getMinor() << "."
     << version.getPatch();
  return os;
}

}  // namespace mlir

// ODS-generated op-interface classes.
#include "shardy/dialect/sdy/ir/op_interface.h.inc"
// ODS-generated op classes.
#define GET_OP_CLASSES
#include "shardy/dialect/sdy/ir/ops.h.inc"

// IWYU pragma: end_exports

#endif  // SHARDY_DIALECT_SDY_IR_DIALECT_H_
