/* Copyright 2026 The MPMD Authors.

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

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_VALIDATENOINFERREDFRAGMENTSPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

class ValidateNoInferredFragmentsPass
    : public impl::ValidateNoInferredFragmentsPassBase<
          ValidateNoInferredFragmentsPass> {
 public:
  using ValidateNoInferredFragmentsPassBase::
      ValidateNoInferredFragmentsPassBase;

 protected:
  void runOnFunc(func::FuncOp func) override {
    func.walk([&](FragmentOp fragment) {
      if (fragment.isUserFragment()) {
        return;
      }

      ArrayAttr inferredByAttr =
          fragment->getAttrOfType<ArrayAttr>(kInferredByAttr);
      if (!inferredByAttr || inferredByAttr.empty()) {
        fragment.emitError() << "Internal error: inferred fragment missing '"
                             << kInferredByAttr << "' attribute";
        return signalPassFailure();
      }

      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "Inferred fragment has not been merged (inferred by ";
      llvm::interleaveComma(inferredByAttr, os, [&](Attribute attr) {
        os << cast<StringAttr>(attr).getValue();
      });
      os << ")";

      // TODO(b/495822074): Remove uniquify exception.
      // TODO(b/513145099): Remove infer_mesh_wrap_meshless_ops exception.
      // TODO(b/513153410): Remove extract_reshards exception.
      if (llvm::any_of(inferredByAttr, [](Attribute attr) {
            StringRef value = cast<StringAttr>(attr).getValue();
            return value == "uniquify" ||
                   value == "infer_mesh_wrap_meshless_ops" ||
                   value == "extract_reshards";
          })) {
        SDY_LOG(WARNING) << msg;
        fragment.emitWarning() << msg;
        return;
      }

      SDY_LOG(WARNING) << msg;
      InFlightDiagnostic diag = failOnInferredFragments
                                    ? fragment.emitError()
                                    : fragment.emitWarning();
      diag << msg;
      if (failOnInferredFragments) {
        signalPassFailure();
      }
    });
  }
};

}  // namespace

}  // namespace mlir::mpmd
