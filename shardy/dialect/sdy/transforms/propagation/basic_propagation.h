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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_BASIC_PROPAGATION_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_BASIC_PROPAGATION_H_

#include <functional>
#include <memory>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_group_map.h"

namespace mlir {
namespace sdy {

// A function that determines in which direction propagation should happen for a
// given op.
using GetDirectionToPropagateFn =
    std::function<PropagationDirection(Operation*)>;

// A function that returns `PropagationDirection::BOTH` for all operations.
PropagationDirection propagateAny(Operation* op);

struct PropagationOptions {
  bool keepShardingRules = false;
  StringRef dumpDirectory = "";
  bool conservativePropagation = false;
  bool debugShardingOrigins = false;
  bool debugEdgeSourceSharding = false;
};

// The implementation class for the basic propagation pass.
//
// Higher strategies in the hierarchy should extend this class, and override one
// or more of the virtual methods that this class provides.
class BasicPropagationPassImpl : public OperationPass<ModuleOp> {
 public:
  using OperationPass<ModuleOp>::OperationPass;

  BasicPropagationPassImpl() = default;
  BasicPropagationPassImpl(const BasicPropagationPassImpl& other)
      : OperationPass<ModuleOp>(other) {}

 protected:
  // Runs propagation on every function in `moduleOp`.
  //
  // This includes propagating the sharding of function outputs to their
  // producing value before this method is called, and propagating the updated
  // sharding back to the function outputs afterwards.
  //
  // The `factorPropagation` determines how we propagate shardings along
  // factors.
  //
  // The `getDirectionToPropagate` determines in which direction propagation
  // should happen on a given operation.
  //
  // NOTE: there is no propagation between call ops and their called functions
  // (e.g. pushing the sharding of an operand in a call op to the function's
  // argument) as we assume the inliner pass was called.
  LogicalResult propagate(
      ModuleOp moduleOp, const SymbolTable& symbolTable,
      const ShardingGroupMap& shardingGroupMap,
      const FactorPropagation& factorPropagation,
      GetDirectionToPropagateFn getDirectionToPropagate = propagateAny);

  // Same as `propagate` above, but uses the strategy in private member
  // `basicFactorPropagation`. Sub-classes should override this method to
  // extend the propagation algorithm and define higher level strategies.
  virtual LogicalResult propagate(
      ModuleOp moduleOp, const SymbolTable& symbolTable,
      const ShardingGroupMap& shardingGroupMap,
      GetDirectionToPropagateFn getDirectionToPropagate = propagateAny);

  const BasicFactorPropagation& getBasicFactorPropagation() const {
    return basicFactorPropagation;
  };

  void runOnOperation() override;

  // Sets the propagation options declared below.
  void setPropagationOptions(const PropagationOptions& options);

  Option<bool> keepShardingRules{
      *this, "keep-sharding-rules",
      llvm::cl::desc("whether to keep existing and created op sharding rules"),
      llvm::cl::init(false)};

  Option<std::string> dumpDirectory{
      *this, "module-dump-directory",
      llvm::cl::desc("where to dump any rewritten modules for debugging"),
      llvm::cl::init("")};

  // TODO(b/347180954): remove conservative propagation once the cost model
  // supports split axes and padding.
  Option<bool> conservativePropagation{
      *this, "conservative-propagation",
      llvm::cl::desc("whether to disallow split axes and non-divisible "
                     "sharding axes during propagation"),
      llvm::cl::init(false)};

  Option<bool> debugShardingOrigins{
      *this, "debug-sharding-origins",
      llvm::cl::desc(
          "whether to save information about the origin of a sharding "
          "on the MLIR module. These would be the shardings on the function "
          "inputs, outputs, sharding constraints and manual computations "
          "before propagation."),
      llvm::cl::init(false)};

  Option<bool> debugEdgeSourceSharding{
      *this, "debug-edge-source-sharding",
      llvm::cl::desc(
          "whether to save information about the edge source of a sharding "
          "on the MLIR module. These are from which operand/result a sharding "
          "was propagated."),
      llvm::cl::init(false)};

 private:
  // This class owns the basic factor propagation strategy.
  BasicFactorPropagation basicFactorPropagation;
};

// Runs the basic sharding propagation algorithm (see
// `BasicPropagationPass`).
std::unique_ptr<Pass> createBasicPropagationPass(
    const PropagationOptions& options);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_BASIC_PROPAGATION_H_
