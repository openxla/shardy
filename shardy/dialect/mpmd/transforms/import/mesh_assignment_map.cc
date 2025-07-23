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

#include "shardy/dialect/mpmd/transforms/import/mesh_assignment_map.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::mpmd {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const UserAssignmentMapOption& option) {
  llvm::interleaveComma(option.value, os, [&](const auto& entry) {
    auto [mesh, stage] = entry.second;
    os << entry.first << "@" << mesh;
    if (stage) {
      os << "/" << *stage;
    }
  });
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const IndexedAssignmentMapOption& option) {
  llvm::interleaveComma(option.value, os, [&](const auto& entry) {
    os << entry.first << "@" << entry.second;
  });
  return os;
}

IndexedAssignmentMap ConvertMeshVectorToMap(
    const std::vector<std::optional<std::string>>& meshes) {
  IndexedAssignmentMap index_to_mesh_map;
  for (auto [index, optionalMesh] : llvm::enumerate(meshes)) {
    if (optionalMesh) {
      index_to_mesh_map.try_emplace(index, *optionalMesh);
    }
  }
  return index_to_mesh_map;
}

}  // namespace mlir::mpmd

namespace llvm::cl {

using ::mlir::mpmd::IndexedAssignmentMap;
using ::mlir::mpmd::IndexedAssignmentMapOption;
using ::mlir::mpmd::UserAssignmentMap;
using ::mlir::mpmd::UserAssignmentMapOption;

//===----------------------------------------------------------------------===//
// UserAssignmentMapOption
//===----------------------------------------------------------------------===//

template class basic_parser<UserAssignmentMapOption>;

bool parser<UserAssignmentMapOption>::parse(Option& opt, StringRef,
                                            StringRef arg,
                                            UserAssignmentMapOption& value) {
  UserAssignmentMap& assignment = value.value;
  StringRef cur;
  while (!arg.empty()) {
    // This allows a redundant comma as the last character.
    std::tie(cur, arg) = arg.split(',');
    cur = cur.trim();
    size_t name_separator_index = cur.find('@');
    if (name_separator_index == StringRef::npos) {
      return opt.error(
          "Assignment must contain '@' denoting a name assigned to a mesh (and "
          "stage), got: " +
          cur);
    }
    StringRef name = cur.take_front(name_separator_index);
    StringRef target = cur.drop_front(name_separator_index + 1);
    size_t mesh_separator_index = target.find('/');
    if (mesh_separator_index == StringRef::npos) {
      assignment[name.str()] = std::make_pair(target, std::nullopt);
    } else {
      StringRef stage_str = target.drop_front(mesh_separator_index + 1);
      int64_t stage;
      if (stage_str.getAsInteger(10, stage)) {
        return opt.error("Failed to parse stage number: " + stage_str);
      }
      assignment[name.str()] =
          std::make_pair(target.take_front(mesh_separator_index), stage);
    }
  }
  return false;
}

void parser<UserAssignmentMapOption>::printOptionDiff(
    const Option& opt, const UserAssignmentMapOption& value,
    const OptVal& defaultValue, size_t globalWidth) const {
  printOptionName(opt, globalWidth);
  outs() << "= " << value;
  if (defaultValue.hasValue()) {
    outs().indent(2) << " (default: " << defaultValue.getValue() << ")";
  }
  outs() << "\n";
}

void parser<UserAssignmentMapOption>::anchor() {}

//===----------------------------------------------------------------------===//
// IndexedAssignmentMapOption
//===----------------------------------------------------------------------===//

template class basic_parser<IndexedAssignmentMapOption>;

bool parser<IndexedAssignmentMapOption>::parse(
    Option& opt, StringRef, StringRef arg, IndexedAssignmentMapOption& value) {
  IndexedAssignmentMap& assignment = value.value;
  StringRef cur;
  while (!arg.empty()) {
    // This allows a redundant comma as the last character.
    std::tie(cur, arg) = arg.split(',');
    cur = cur.trim();
    size_t name_separator_index = cur.find('@');
    if (name_separator_index == StringRef::npos) {
      return opt.error(
          "Assignment must contain '@' denoting a name assigned to a mesh, "
          "got: " +
          cur);
    }
    StringRef index_str = cur.take_front(name_separator_index);
    StringRef target = cur.drop_front(name_separator_index + 1);
    int64_t index;
    if (index_str.getAsInteger(10, index)) {
      return opt.error("Failed to parse index: " + index_str);
    }

    assignment[index] = target;
  }
  return false;
}

void parser<IndexedAssignmentMapOption>::printOptionDiff(
    const Option& opt, const IndexedAssignmentMapOption& value,
    const OptVal& defaultValue, size_t globalWidth) const {
  printOptionName(opt, globalWidth);
  outs() << "= " << value;
  if (defaultValue.hasValue()) {
    outs().indent(2) << " (default: " << defaultValue.getValue() << ")";
  }
  outs() << "\n";
}

void parser<IndexedAssignmentMapOption>::anchor() {}

}  // namespace llvm::cl
