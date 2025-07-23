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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESH_ASSIGNMENT_MAP_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESH_ASSIGNMENT_MAP_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::mpmd {

using MeshStageAssignment = std::pair<std::string, std::optional<int64_t>>;

// A user-defined mapping between names (of computations and tensors) and mesh
// names and optionally stage ids. E.g., n -> {m, s} means that the name n is
// assigned to mesh m, stage s; n -> {m, nullopt} means that n is assigned to
// mesh m, but to no particular stage of that mesh.
//
// Note: `std::map` is used to ensure stable iteration order for the MPMD
// compilation cache key. We replace the default comparator with `std::less<>`
// so we can lookup using `std::string_view`.
using UserAssignmentMap =
    std::map<std::string, MeshStageAssignment, std::less<>>;

// A `UserAssignmentMap` option with a custom parser/printer.
struct UserAssignmentMapOption {
  UserAssignmentMap value;
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const UserAssignmentMapOption& option);

// A user-defined mapping between function input/output indices and mesh names.
// E.g., i -> m means that the ith input/output  is assigned to mesh m.
//
// Note: we use a map instead of a vector here because it's used in two ways:
// 1. In testing, we need to support passing a comma separated list of mappings
//    in the format "index@mesh_name".
// 2. In production, we need to support passing a vector of
//    `std::optional<std::string>` from the Python code.
// It is possible to convert the vector to a map but not vice versa.
using IndexedAssignmentMap = llvm::DenseMap<int, std::string>;

// A `IndexedAssignmentMap` option with a custom parser/printer.
struct IndexedAssignmentMapOption {
  IndexedAssignmentMap value;
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const IndexedAssignmentMapOption& option);

// Converts a vector of optional mesh names to a map of index (in the vector) to
// mesh name.
IndexedAssignmentMap ConvertMeshVectorToMap(
    const std::vector<std::optional<std::string>>& meshes);

}  // namespace mlir::mpmd

namespace llvm::cl {

extern template class basic_parser<mlir::mpmd::UserAssignmentMapOption>;
extern template class basic_parser<mlir::mpmd::IndexedAssignmentMapOption>;

template <>
class parser<mlir::mpmd::UserAssignmentMapOption>
    : public basic_parser<mlir::mpmd::UserAssignmentMapOption> {
 public:
  parser(Option& opt) : basic_parser(opt) {}
  bool parse(Option& opt, StringRef argName, StringRef arg,
             mlir::mpmd::UserAssignmentMapOption& value);
  StringRef getValueName() const override { return "user-assignment-map"; }
  void printOptionDiff(const Option& opt,
                       const mlir::mpmd::UserAssignmentMapOption& value,
                       const OptVal& defaultValue, size_t globalWidth) const;
  void anchor() override;
};

template <>
class parser<mlir::mpmd::IndexedAssignmentMapOption>
    : public basic_parser<mlir::mpmd::IndexedAssignmentMapOption> {
 public:
  parser(Option& opt) : basic_parser(opt) {}
  bool parse(Option& opt, StringRef argName, StringRef arg,
             mlir::mpmd::IndexedAssignmentMapOption& value);
  StringRef getValueName() const override { return "indexed-assignment-map"; }
  void printOptionDiff(const Option& opt,
                       const mlir::mpmd::IndexedAssignmentMapOption& value,
                       const OptVal& defaultValue, size_t globalWidth) const;
  void anchor() override;
};

}  // namespace llvm::cl

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESH_ASSIGNMENT_MAP_H_
