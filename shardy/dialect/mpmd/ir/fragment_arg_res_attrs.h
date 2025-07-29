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

#ifndef SHARDY_DIALECT_MPMD_IR_FRAGMENT_ARG_RES_ATTRS_H_
#define SHARDY_DIALECT_MPMD_IR_FRAGMENT_ARG_RES_ATTRS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"

namespace mlir::mpmd {

// Attribute to store arg attrs for ops without this built in.
constexpr StringRef kArgAttrName = "arg_attrs";
// Attribute to store res attrs for ops without this built in.
constexpr StringRef kResAttrName = "res_attrs";

// Get the existing arg attr if it exists, or create a new one of the length of
// the operands.
inline SmallVector<Attribute> GetArgAttrsOrCreateDefault(Operation* op) {
  return op->hasAttr(kArgAttrName)
             ? llvm::to_vector(
                   op->getAttrOfType<ArrayAttr>(kArgAttrName).getValue())
             : SmallVector<Attribute>(
                   op->getNumOperands(),
                   DictionaryAttr::get(op->getContext(), {}));
}

// Retrieve the arg attr at `index`.
inline Attribute GetArgAttr(Operation* op, int index, StringRef name) {
  if (auto attr = op->getAttrOfType<ArrayAttr>(kArgAttrName)) {
    return cast<DictionaryAttr>(attr[index]).get(name);
  }
  return nullptr;
}

// Update the arg attrs Attribute.
inline void SetArgAttrs(Operation* op, ArrayRef<Attribute> arg_attrs) {
  op->setAttr(kArgAttrName, ArrayAttr::get(op->getContext(), arg_attrs));
}

// Get the existing res attr if it exists, or create a new one of the length of
// the results.
inline SmallVector<Attribute> GetResAttrsOrCreateDefault(Operation* op) {
  return op->hasAttr(kResAttrName)
             ? llvm::to_vector(
                   op->getAttrOfType<ArrayAttr>(kResAttrName).getValue())
             : SmallVector<Attribute>(
                   op->getNumResults(),
                   DictionaryAttr::get(op->getContext(), {}));
}

// Retrieve the res attr at `index`.
inline Attribute GetResAttr(Operation* op, int index, StringRef name) {
  if (auto attr = op->getAttrOfType<ArrayAttr>(kResAttrName)) {
    return cast<DictionaryAttr>(attr[index]).get(name);
  }
  return nullptr;
}

// Update the res attrs Attribute.
inline void SetResAttrs(Operation* op, ArrayRef<Attribute> res_attrs) {
  op->setAttr(kResAttrName, ArrayAttr::get(op->getContext(), res_attrs));
}

// Updates `base_dict_attr` with the new value in place. Crashes if the value
// already exists and `insert_if_not_present = false`.
inline void InsertAttr(Attribute& base_dict_attr, StringRef name,
                       Attribute value, bool insert_if_not_present = false) {
  NamedAttrList attributes(cast<DictionaryAttr>(base_dict_attr));
  Attribute old_value = attributes.set(name, value);
  if (!insert_if_not_present) {
    SDY_CHECK(!old_value);
  }
  base_dict_attr = attributes.getDictionary(base_dict_attr.getContext());
}

// Removes in-place the attribute with the given name from the base dictionary.
// Returns true if the attribute was removed.
inline bool RemoveAttr(Attribute& base_dict_attr, StringRef name) {
  NamedAttrList attributes(cast<DictionaryAttr>(base_dict_attr));
  bool removed = attributes.erase(name) != nullptr;
  base_dict_attr = attributes.getDictionary(base_dict_attr.getContext());
  return removed;
}

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_IR_FRAGMENT_ARG_RES_ATTRS_H_
