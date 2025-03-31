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
#ifndef SHARDY_DIALECT_SDY_IR_BYTECODE_H_
#define SHARDY_DIALECT_SDY_IR_BYTECODE_H_

#include <cstdint>
#include <optional>

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sdy {

class SdyDialect;

namespace detail {
void addBytecodeInterface(SdyDialect *dialect);
}  // namespace detail

/// Write a VarInt and a flag packed together.
void writeOptionalVarInt(DialectBytecodeWriter &writer,
                         std::optional<uint64_t> value);
/// Parse a variable length encoded integer whose low bit is used to encode an
/// unrelated flag, i.e: `(integerValue << 1) | (flag ? 1 : 0)`.
LogicalResult readOptionalVarInt(DialectBytecodeReader &reader,
                                 std::optional<uint64_t> &result);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_IR_BYTECODE_H_
