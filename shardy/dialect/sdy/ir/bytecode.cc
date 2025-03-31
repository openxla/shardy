/* Copyright 2025 The Shardy Authors.

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
#include "shardy/dialect/sdy/ir/bytecode.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"

using namespace mlir;
using namespace mlir::sdy;

//===----------------------------------------------------------------------===//
// Debug Trace Helpers
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sdy-bytecode"

#define LOG_NOT_IMPLEMENTED(typeOrAttr)                                       \
  DEBUG_WITH_TYPE("sdy-bytecode", llvm::errs()                                \
                                      << "***Not Implemented: " << typeOrAttr \
                                      << " " << LLVM_PRETTY_FUNCTION << '\n')

namespace {

#include "shardy/dialect/sdy/ir/bytecode.cc.inc"

/// This class implements the bytecode interface for the SDY dialect.
struct SdyDialectBytecodeInterface : public BytecodeDialectInterface {
  SdyDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    auto result = ::writeAttribute(attr, writer);
    if (failed(result)) {
      LOG_NOT_IMPLEMENTED(attr);
    }
    return result;
  }

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    auto result = ::writeType(type, writer);
    if (failed(result)) {
      LOG_NOT_IMPLEMENTED(type);
    }
    return result;
  }

  std::unique_ptr<DialectVersion> readVersion(
      DialectBytecodeReader &reader) const override {
    uint64_t major, minor, patch;
    if (failed(reader.readVarInt(major)) || failed(reader.readVarInt(minor)) ||
        failed(reader.readVarInt(patch)))
      return nullptr;

    auto version = std::make_unique<SdyDialectVersion>(
        /*major=*/major, /*minor=*/minor, /*patch=*/patch);
    if (version && SdyDialectVersion::getCurrentVersion() < *version) {
      mlir::emitWarning(mlir::UnknownLoc::get(getContext()))
          << "reading newer dialect than supported";
      return nullptr;
    }
    return version;
  }

  void writeVersion(DialectBytecodeWriter &writer) const override {
    if (auto *dialect = dyn_cast<SdyDialect>(getDialect())) {
      FailureOr<const DialectVersion *> versionOrFailed =
          writer.getDialectVersion(dialect->getNamespace());
      if (failed(versionOrFailed)) {
        return;
      }
      const auto *version =
          static_cast<const SdyDialectVersion *>(*versionOrFailed);
      writer.writeVarInt(static_cast<uint64_t>(version->getMajor()));
      writer.writeVarInt(static_cast<uint64_t>(version->getMinor()));
      writer.writeVarInt(static_cast<uint64_t>(version->getPatch()));
    }
  }

  LogicalResult upgradeFromVersion(Operation *topLevelOp,
                                   const DialectVersion &version) const final {
    DEBUG_WITH_TYPE("sdy-bytecode",
                    llvm::dbgs()
                        << "upgrading from version "
                        << static_cast<const SdyDialectVersion &>(version)
                        << "\n");
    return success();
  }
};
}  // namespace

namespace mlir {
namespace sdy {

void writeOptionalVarInt(DialectBytecodeWriter &writer,
                         std::optional<uint64_t> value) {
  writer.writeVarIntWithFlag(value.value_or(0), /*flag=*/value.has_value());
}

LogicalResult readOptionalVarInt(DialectBytecodeReader &reader,
                                 std::optional<uint64_t> &result) {
  uint64_t resultValue;
  bool flag;
  if (failed(reader.readVarIntWithFlag(resultValue, flag))) return failure();
  if (flag) {
    result = resultValue;
  }
  return success();
}

}  // namespace sdy
}  // namespace mlir

void sdy::detail::addBytecodeInterface(SdyDialect *dialect) {
  dialect->addInterfaces<SdyDialectBytecodeInterface>();
}
