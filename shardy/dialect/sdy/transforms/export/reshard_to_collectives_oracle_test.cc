/* Copyright 2026 The Shardy Authors.

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

// Compares `-sdy-reshard-to-collectives` against an exact shortest-path oracle.
//
// For a small mesh, every sharding of a tensor is a node of a graph, and every
// collective the pass can emit (all-slice, all-gather, single-parameter
// all-to-all, collective-permute) is an edge weighted by the number of
// elements each device receives. Dijkstra gives the cheapest sequence of
// collectives between any two shardings, which we compare against the
// sequence the pass actually emits, for all pairs of shardings.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

struct MeshAxis {
  std::string name;
  int64_t size;
};

// An indivisible piece of a mesh axis. An axis of size p^k (p prime) is split
// into k atoms of size p, ordered from major to minor; any other axis is a
// single atom.
struct Atom {
  int axis;
  int64_t preSize;
  int64_t size;
};

// Atoms sharding each tensor dimension, from major to minor.
using State = std::vector<std::vector<int>>;

enum class MoveKind { kAllSlice, kAllGather, kAllToAll, kCollectivePermute };

const char* moveName(MoveKind kind) {
  switch (kind) {
    case MoveKind::kAllSlice:
      return "all_slice";
    case MoveKind::kAllGather:
      return "all_gather";
    case MoveKind::kAllToAll:
      return "all_to_all";
    case MoveKind::kCollectivePermute:
      return "collective_permute";
  }
  return "";
}

// Cost of a sequence of collectives: elements received per device, then the
// number of collectives.
struct Cost {
  int64_t elems = 0;
  int64_t ops = 0;

  bool operator<(const Cost& other) const {
    return std::tie(elems, ops) < std::tie(other.elems, other.ops);
  }
  bool operator==(const Cost& other) const {
    return elems == other.elems && ops == other.ops;
  }
  Cost operator+(const Cost& other) const {
    return {elems + other.elems, ops + other.ops};
  }
};

struct Edge {
  int to;
  Cost cost;
  MoveKind kind;
};

struct Step {
  MoveKind kind;
  int state;
};

std::optional<int64_t> primeOfPrimePower(int64_t n) {
  for (int64_t p = 2; p * p <= n; ++p) {
    if (n % p == 0) {
      while (n % p == 0) n /= p;
      return n == 1 ? std::optional<int64_t>(p) : std::nullopt;
    }
  }
  return n > 1 ? std::optional<int64_t>(n) : std::nullopt;
}

// Exact shortest-path oracle over all shardings of a tensor on a mesh.
class ReshardOracle {
 public:
  ReshardOracle(std::vector<MeshAxis> axes, std::vector<int64_t> shape)
      : axes_(std::move(axes)), shape_(std::move(shape)) {
    for (auto [axisIndex, axis] : llvm::enumerate(axes_)) {
      std::optional<int64_t> prime = primeOfPrimePower(axis.size);
      int64_t atomSize = prime.value_or(axis.size);
      for (int64_t pre = 1; pre < axis.size; pre *= atomSize) {
        atoms_.push_back({static_cast<int>(axisIndex), pre, atomSize});
      }
    }
    assert(atoms_.size() < 32 &&
           "ReshardOracle bitmask supports at most 31 atoms");
    globalElems_ = 1;
    for (int64_t dimSize : shape_) globalElems_ *= dimSize;
    State cur(shape_.size());
    enumerateStates(0, 0, cur);
    buildEdges();
  }

  int numStates() const { return states_.size(); }
  int rank() const { return shape_.size(); }
  const State& state(int i) const { return states_[i]; }
  const std::vector<Atom>& atoms() const { return atoms_; }
  const std::vector<MeshAxis>& axes() const { return axes_; }
  const std::vector<int64_t>& shape() const { return shape_; }

  std::optional<int> indexOf(const State& s) const {
    auto it = index_.find(key(s));
    if (it == index_.end()) return std::nullopt;
    return it->second;
  }

  int64_t localElems(int s) const { return localElems_[s]; }

  // Dijkstra from `src`; fills `parent` with the last step of each best path.
  std::vector<Cost> shortestFrom(int src, std::vector<Step>* parent) const {
    constexpr int64_t kInf = std::numeric_limits<int64_t>::max();
    std::vector<Cost> dist(numStates(), Cost{kInf, kInf});
    parent->assign(numStates(), Step{MoveKind::kAllSlice, -1});
    using Item = std::pair<Cost, int>;
    auto greater = [](const Item& a, const Item& b) {
      return b.first < a.first;
    };
    std::priority_queue<Item, std::vector<Item>, decltype(greater)> queue(
        greater);
    dist[src] = Cost{};
    queue.push({dist[src], src});
    while (!queue.empty()) {
      auto [d, s] = queue.top();
      queue.pop();
      if (dist[s] < d) continue;
      for (const Edge& e : edges_[s]) {
        Cost nd = d + e.cost;
        if (nd < dist[e.to]) {
          dist[e.to] = nd;
          (*parent)[e.to] = Step{e.kind, s};
          queue.push({nd, e.to});
        }
      }
    }
    return dist;
  }

  // Information-theoretic lower bound on elements received per device: the
  // part of a device's target shard it does not already hold.
  int64_t lowerBound(int src, int tgt) const {
    int64_t numDevices = 1;
    for (const Atom& atom : atoms_) numDevices *= atom.size;
    std::vector<int64_t> coord(atoms_.size(), 0);
    int64_t best = 0;
    for (int64_t device = 0; device < numDevices; ++device) {
      int64_t rem = device;
      for (auto [i, atom] : llvm::enumerate(atoms_)) {
        coord[i] = rem % atom.size;
        rem /= atom.size;
      }
      int64_t tgtElems = 1, overlap = 1;
      for (int d = 0; d < rank(); ++d) {
        auto [srcBegin, srcLen] = interval(states_[src][d], d, coord);
        auto [tgtBegin, tgtLen] = interval(states_[tgt][d], d, coord);
        tgtElems *= tgtLen;
        overlap *= std::max<int64_t>(
            0, std::min(srcBegin + srcLen, tgtBegin + tgtLen) -
                   std::max(srcBegin, tgtBegin));
      }
      best = std::max(best, tgtElems - overlap);
    }
    return best;
  }

  // Prints `s` in sdy syntax, merging consecutive atoms of the same axis.
  std::string toString(const State& s) const {
    std::string result = "[";
    for (auto [d, dimAtoms] : llvm::enumerate(s)) {
      if (d > 0) result += ", ";
      result += "{";
      for (size_t i = 0; i < dimAtoms.size();) {
        const Atom& first = atoms_[dimAtoms[i]];
        int64_t size = first.size;
        size_t j = i + 1;
        while (j < dimAtoms.size() && atoms_[dimAtoms[j]].axis == first.axis &&
               atoms_[dimAtoms[j]].preSize == first.preSize * size) {
          size *= atoms_[dimAtoms[j]].size;
          ++j;
        }
        if (i > 0) result += ", ";
        const MeshAxis& axis = axes_[first.axis];
        result += "\"" + axis.name + "\"";
        if (size != axis.size) {
          result +=
              ":(" + std::to_string(first.preSize) + ")" + std::to_string(size);
        }
        i = j;
      }
      result += "}";
    }
    return result + "]";
  }

  // Maps an axis ref to its atoms, or nullopt if it splits an atom.
  std::optional<std::vector<int>> atomsOf(AxisRefAttr axisRef) const {
    auto axisIt = llvm::find_if(
        axes_, [&](const MeshAxis& a) { return a.name == axisRef.getName(); });
    if (axisIt == axes_.end()) return std::nullopt;
    int axisIndex = axisIt - axes_.begin();
    int64_t pre = 1, size = axisIt->size;
    if (SubAxisInfoAttr info = axisRef.getSubAxisInfo()) {
      pre = info.getPreSize();
      size = info.getSize();
    }
    std::vector<int> result;
    int64_t covered = 1;
    for (auto [i, atom] : llvm::enumerate(atoms_)) {
      if (atom.axis != axisIndex || atom.preSize < pre ||
          atom.preSize >= pre * size) {
        continue;
      }
      if (atom.preSize != pre * covered) return std::nullopt;
      result.push_back(i);
      covered *= atom.size;
    }
    if (covered != size) return std::nullopt;
    return result;
  }

  std::optional<int> indexOf(TensorShardingAttr sharding) const {
    State s;
    for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
      std::vector<int>& dimAtoms = s.emplace_back();
      for (AxisRefAttr axisRef : dimSharding.getAxes()) {
        std::optional<std::vector<int>> refAtoms = atomsOf(axisRef);
        if (!refAtoms) return std::nullopt;
        dimAtoms.insert(dimAtoms.end(), refAtoms->begin(), refAtoms->end());
      }
    }
    return indexOf(s);
  }

  int64_t atomsSize(llvm::ArrayRef<int> atomIndices) const {
    int64_t size = 1;
    for (int i : atomIndices) size *= atoms_[i].size;
    return size;
  }

  // Per-device elements received by an all-to-all over `groupSize` devices.
  static int64_t allToAllElems(int64_t local, int64_t groupSize) {
    return local / groupSize * (groupSize - 1);
  }

 private:
  static std::string key(const State& s) {
    std::string k;
    for (const std::vector<int>& dimAtoms : s) {
      for (int a : dimAtoms) k.push_back(static_cast<char>(a + 1));
      k.push_back('\0');
    }
    return k;
  }

  // Global [begin, length) of the shard held by the device at `coord` along
  // dimension `d`, when that dimension is sharded by `dimAtoms`.
  std::pair<int64_t, int64_t> interval(
      const std::vector<int>& dimAtoms, int d,
      const std::vector<int64_t>& coord) const {
    int64_t index = 0, numShards = 1;
    for (int a : dimAtoms) {
      index = index * atoms_[a].size + coord[a];
      numShards *= atoms_[a].size;
    }
    int64_t len = shape_[d] / numShards;
    return {index * len, len};
  }

  void enumerateStates(int dim, uint32_t used, State& cur) {
    if (dim == rank()) {
      index_[key(cur)] = states_.size();
      states_.push_back(cur);
      return;
    }
    enumerateStates(dim + 1, used, cur);
    for (int a = 0; a < static_cast<int>(atoms_.size()); ++a) {
      if (used & (1u << a)) continue;
      cur[dim].push_back(a);
      enumerateStates(dim, used | (1u << a), cur);
      cur[dim].pop_back();
    }
  }

  uint32_t usedMask(const State& s) const {
    uint32_t mask = 0;
    for (const std::vector<int>& dimAtoms : s) {
      for (int a : dimAtoms) mask |= 1u << a;
    }
    return mask;
  }

  // Calls `fn` on every state obtained by appending unused atoms to `s`.
  void forEachExtension(int dim, uint32_t used, State& cur,
                        const std::function<void(const State&)>& fn) const {
    if (dim == rank()) {
      fn(cur);
      return;
    }
    forEachExtension(dim + 1, used, cur, fn);
    for (int a = 0; a < static_cast<int>(atoms_.size()); ++a) {
      if (used & (1u << a)) continue;
      cur[dim].push_back(a);
      forEachExtension(dim, used | (1u << a), cur, fn);
      cur[dim].pop_back();
    }
  }

  void buildEdges() {
    int n = numStates();
    localElems_.resize(n);
    std::vector<std::vector<int64_t>> sizesPerDim(n);
    std::map<std::vector<int64_t>, std::vector<int>> sameSizes;
    for (int s = 0; s < n; ++s) {
      int64_t numShards = 1;
      for (const std::vector<int>& dimAtoms : states_[s]) {
        int64_t dimShards = atomsSize(dimAtoms);
        sizesPerDim[s].push_back(dimShards);
        numShards *= dimShards;
      }
      localElems_[s] = globalElems_ / numShards;
      sameSizes[sizesPerDim[s]].push_back(s);
    }
    edges_.resize(n);
    for (int s = 0; s < n; ++s) {
      int64_t local = localElems_[s];
      // All-slice to any extension, and the reverse all-gather.
      State cur = states_[s];
      forEachExtension(0, usedMask(cur), cur, [&](const State& t) {
        int ti = index_.at(key(t));
        if (ti == s) return;
        edges_[s].push_back({ti, Cost{0, 1}, MoveKind::kAllSlice});
        edges_[ti].push_back(
            {s, Cost{local - localElems_[ti], 1}, MoveKind::kAllGather});
      });
      // All-to-all of a suffix of one dimension to the end of another.
      for (int src = 0; src < rank(); ++src) {
        const std::vector<int>& srcAtoms = states_[s][src];
        for (size_t start = 0; start < srcAtoms.size(); ++start) {
          llvm::ArrayRef<int> moved =
              llvm::ArrayRef(srcAtoms).drop_front(start);
          for (int tgt = 0; tgt < rank(); ++tgt) {
            if (tgt == src) continue;
            State t = states_[s];
            t[src].resize(start);
            t[tgt].insert(t[tgt].end(), moved.begin(), moved.end());
            edges_[s].push_back(
                {index_.at(key(t)),
                 Cost{allToAllElems(local, atomsSize(moved)), 1},
                 MoveKind::kAllToAll});
          }
        }
      }
      // Collective-permute to any state with the same per-dimension sizes.
      for (int t : sameSizes[sizesPerDim[s]]) {
        if (t != s) {
          edges_[s].push_back(
              {t, Cost{local, 1}, MoveKind::kCollectivePermute});
        }
      }
    }
  }

  std::vector<MeshAxis> axes_;
  std::vector<int64_t> shape_;
  std::vector<Atom> atoms_;
  int64_t globalElems_;
  std::vector<State> states_;
  std::unordered_map<std::string, int> index_;
  std::vector<int64_t> localElems_;
  std::vector<std::vector<Edge>> edges_;
};

// Result of running the pass on one reshard.
struct PassPlan {
  std::vector<Step> steps;
  Cost cost;
  std::string error;  // Empty iff the plan is valid and reaches the target.
};

std::string tensorType(llvm::ArrayRef<int64_t> shape) {
  std::string result = "tensor<";
  for (int64_t dimSize : shape) result += std::to_string(dimSize) + "x";
  return result + "f32>";
}

bool validateCollectiveStep(Operation& op, const ReshardOracle& oracle,
                            const State& from, const State& to, int64_t local,
                            int64_t nextLocal, MoveKind& kind, int64_t& elems,
                            std::string& error) {
  auto isPrefix = [](const State& a, const State& b) {
    for (auto [x, y] : llvm::zip_equal(a, b)) {
      if (x.size() > y.size() || !std::equal(x.begin(), x.end(), y.begin())) {
        return false;
      }
    }
    return true;
  };
  elems = 0;
  bool valid = true;
  if (isa<AllSliceOp>(op)) {
    kind = MoveKind::kAllSlice;
    valid = isPrefix(from, to);
  } else if (isa<AllGatherOp>(op)) {
    kind = MoveKind::kAllGather;
    valid = isPrefix(to, from);
    elems = nextLocal - local;
  } else if (auto allToAll = dyn_cast<AllToAllOp>(op)) {
    kind = MoveKind::kAllToAll;
    State expected = from;
    int64_t groupSize = 1;
    for (AllToAllParamAttr param : allToAll.getParams()) {
      std::vector<int> moved;
      for (AxisRefAttr axisRef : param.getAxes()) {
        std::optional<std::vector<int>> refAtoms = oracle.atomsOf(axisRef);
        if (!refAtoms) {
          valid = false;
          break;
        }
        moved.insert(moved.end(), refAtoms->begin(), refAtoms->end());
      }
      std::vector<int>& srcAtoms = expected[param.getSrcDim()];
      if (!valid || moved.size() > srcAtoms.size() ||
          !std::equal(moved.begin(), moved.end(),
                      srcAtoms.end() - moved.size())) {
        valid = false;
        break;
      }
      srcAtoms.resize(srcAtoms.size() - moved.size());
      llvm::append_range(expected[param.getTgtDim()], moved);
      groupSize *= oracle.atomsSize(moved);
    }
    valid = valid && expected == to;
    elems = ReshardOracle::allToAllElems(local, groupSize);
  } else if (isa<CollectivePermuteOp>(op)) {
    kind = MoveKind::kCollectivePermute;
    for (auto [x, y] : llvm::zip_equal(from, to)) {
      valid &= oracle.atomsSize(x) == oracle.atomsSize(y);
    }
    elems = local;
  } else {
    error = "unsupported collective";
    return false;
  }
  if (!valid) {
    error = std::string("invalid ") + moveName(kind);
    return false;
  }
  return true;
}

// Runs the pass on one function per (src, tgt) pair using `shape` and decodes
// the emitted collectives into oracle states. When `checkPaddingInvariants` is
// true, checks that non-replicated sharded dimension transitions preserve
// padded dimension sizes on uneven shapes.
std::vector<PassPlan> runPassWithShape(
    MLIRContext& context, const ReshardOracle& oracle,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<std::pair<int, int>> pairs,
    bool checkPaddingInvariants = false) {
  std::string type = tensorType(shape);
  std::string program = "sdy.mesh @mesh = <[";
  for (auto [i, axis] : llvm::enumerate(oracle.axes())) {
    if (i > 0) program += ", ";
    program += "\"" + axis.name + "\"=" + std::to_string(axis.size);
  }
  program += "]>\n";
  for (auto [i, pair] : llvm::enumerate(pairs)) {
    program += "func.func @f" + std::to_string(i) + "(%arg0: " + type +
               " {sdy.sharding = #sdy.sharding<@mesh, " +
               oracle.toString(oracle.state(pair.first)) + ">}) -> " + type +
               " {\n  %0 = sdy.reshard %arg0 <@mesh, " +
               oracle.toString(oracle.state(pair.second)) + "> : " + type +
               "\n  return %0 : " + type + "\n}\n";
  }
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  if (!module) {
    ADD_FAILURE() << "Failed to parse generated MLIR program";
    return std::vector<PassPlan>(pairs.size(),
                                 PassPlan{{}, {}, "parse failed"});
  }
  std::string diagnostics;
  LogicalResult passResult = failure();
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diag) {
      diagnostics += diag.str() + "\n";
      return success();
    });
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(createReshardToCollectivesPass());
    passResult = pm.run(*module);
  }
  if (failed(passResult)) {
    if (pairs.size() == 1) {
      return {PassPlan{{}, {}, "pass failed: " + diagnostics}};
    }
    // Binary-split the batch to isolate failing pairs in O(k log N) runs.
    size_t mid = pairs.size() / 2;
    std::vector<PassPlan> plans = runPassWithShape(
        context, oracle, shape, pairs.take_front(mid), checkPaddingInvariants);
    llvm::append_range(
        plans, runPassWithShape(context, oracle, shape, pairs.drop_front(mid),
                                checkPaddingInvariants));
    return plans;
  }

  std::vector<PassPlan> plans(pairs.size());
  int funcIndex = 0;
  for (func::FuncOp funcOp : module->getOps<func::FuncOp>()) {
    PassPlan& plan = plans[funcIndex];
    auto [src, tgt] = pairs[funcIndex++];
    int cur = src;
    Value curValue = funcOp.getArgument(0);
    for (Operation& op : funcOp.getBody().front()) {
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        if (returnOp.getOperand(0) != curValue) plan.error = "bad return";
        break;
      }
      auto collective = dyn_cast<CollectiveOpInterface>(op);
      if (!collective || op.getOperand(0) != curValue) {
        plan.error = "unexpected op";
        break;
      }
      std::optional<int> next = oracle.indexOf(collective.getOutSharding());
      if (!next) {
        plan.error = "unrepresentable sharding";
        break;
      }
      const State& from = oracle.state(cur);
      const State& to = oracle.state(*next);
      MoveKind kind;
      int64_t elems = 0;
      if (!validateCollectiveStep(op, oracle, from, to, oracle.localElems(cur),
                                  oracle.localElems(*next), kind, elems,
                                  plan.error)) {
        break;
      }
      if (checkPaddingInvariants) {
        for (int d = 0; d < oracle.rank(); ++d) {
          int64_t prevShards = oracle.atomsSize(from[d]);
          int64_t nextShards = oracle.atomsSize(to[d]);
          if (prevShards > 1 && nextShards > 1 &&
              llvm::alignTo(shape[d], prevShards) !=
                  llvm::alignTo(shape[d], nextShards)) {
            plan.error = "padded size mismatch on dim " + std::to_string(d);
            break;
          }
        }
        if (!plan.error.empty()) {
          break;
        }
      }
      plan.steps.push_back({kind, *next});
      plan.cost = plan.cost + Cost{elems, 1};
      cur = *next;
      curValue = op.getResult(0);
    }
    if (plan.error.empty() && cur != tgt) plan.error = "wrong final sharding";
  }
  return plans;
}

std::vector<PassPlan> runPass(MLIRContext& context, const ReshardOracle& oracle,
                              llvm::ArrayRef<std::pair<int, int>> pairs) {
  return runPassWithShape(context, oracle, oracle.shape(), pairs);
}

struct SweepStats {
  int64_t pairs = 0;
  // Pairs where the pass fails or emits an invalid or incomplete sequence.
  int64_t numErrors = 0;
  std::vector<std::string> errors;
  int64_t subOptimalElems = 0;
  int64_t subOptimalOps = 0;
  int64_t totalPassElems = 0;
  int64_t totalOracleElems = 0;
  double maxRatio = 1;
  // Histogram of greedy / oracle elements for bytes-suboptimal pairs.
  std::map<std::string, int64_t> ratioHistogram;
};

struct Counterexample {
  int src, tgt;
  Cost greedy, oracle;
  std::vector<Step> greedySteps, oracleSteps;
};

std::string ratioBucket(double ratio) {
  if (ratio <= 1.25) return "(1, 1.25]";
  if (ratio <= 1.5) return "(1.25, 1.5]";
  if (ratio <= 2) return "(1.5, 2]";
  if (ratio <= 4) return "(2, 4]";
  return "> 4";
}

class ReshardOracleTest : public ::testing::Test {
 protected:
  void SetUp() override { loadAllRequiredDialects(&context); }

  // Runs the pass on every (src, tgt) pair with src in `srcs`, checks it is
  // correct and never beats the oracle, and returns stats.
  SweepStats sweep(const ReshardOracle& oracle, const std::vector<int>& srcs,
                   std::vector<Counterexample>* counterexamples,
                   const std::string& name) {
    SweepStats stats;
    // Optionally dumps per-pair costs, to diff two versions of the pass.
    std::unique_ptr<std::ofstream> dump;
    if (const char* dir = std::getenv("RESHARD_ORACLE_DUMP_DIR")) {
      dump = std::make_unique<std::ofstream>(std::string(dir) + "/" + name +
                                             ".tsv");
    }
    constexpr int kBatch = 4096;
    std::vector<std::pair<int, int>> batch;
    std::vector<std::vector<Cost>> dists;
    std::vector<std::vector<Step>> parents;
    std::unordered_map<int, int> srcSlot;
    auto flush = [&]() {
      std::vector<PassPlan> plans = runPass(context, oracle, batch);
      for (auto [pair, plan] : llvm::zip_equal(batch, plans)) {
        auto [src, tgt] = pair;
        const Cost& best = dists[srcSlot[src]][tgt];
        if (!plan.error.empty()) {
          if (stats.errors.size() < 10) {
            stats.errors.push_back(plan.error + ": " +
                                   oracle.toString(oracle.state(src)) + " -> " +
                                   oracle.toString(oracle.state(tgt)));
          }
          ++stats.numErrors;
          continue;
        }
        // The oracle is exact, so the pass can never be cheaper.
        ASSERT_FALSE(plan.cost < best)
            << oracle.toString(oracle.state(src)) << " -> "
            << oracle.toString(oracle.state(tgt));
        ASSERT_GE(best.elems, oracle.lowerBound(src, tgt));
        ++stats.pairs;
        stats.totalPassElems += plan.cost.elems;
        stats.totalOracleElems += best.elems;
        if (dump) {
          *dump << oracle.toString(oracle.state(src)) << " -> "
                << oracle.toString(oracle.state(tgt)) << "\t" << plan.cost.elems
                << "\t" << plan.cost.ops << "\t" << best.elems << "\t"
                << best.ops << "\n";
        }
        if (plan.cost.elems > best.elems) {
          ++stats.subOptimalElems;
          double ratio =
              best.elems == 0
                  ? 1e9
                  : static_cast<double>(plan.cost.elems) / best.elems;
          stats.maxRatio = std::max(stats.maxRatio, ratio);
          ++stats.ratioHistogram[ratioBucket(ratio)];
        } else if (plan.cost.ops > best.ops) {
          ++stats.subOptimalOps;
        }
        if (counterexamples && best < plan.cost) {
          std::vector<Step> oracleSteps;
          for (int s = tgt; s != src;) {
            const Step& step = parents[srcSlot[src]][s];
            oracleSteps.push_back({step.kind, s});
            s = step.state;
          }
          std::reverse(oracleSteps.begin(), oracleSteps.end());
          counterexamples->push_back(
              {src, tgt, plan.cost, best, plan.steps, oracleSteps});
        }
      }
      batch.clear();
      dists.clear();
      parents.clear();
      srcSlot.clear();
    };
    for (int src : srcs) {
      srcSlot[src] = dists.size();
      parents.emplace_back();
      dists.push_back(oracle.shortestFrom(src, &parents.back()));
      for (int tgt = 0; tgt < oracle.numStates(); ++tgt) {
        if (tgt != src) batch.push_back({src, tgt});
      }
      if (batch.size() >= kBatch) flush();
      if (HasFatalFailure()) return stats;
    }
    if (!batch.empty()) flush();
    return stats;
  }

  void verifyNonDivisibleAllPairs(const std::vector<MeshAxis>& axes,
                                  const std::vector<int64_t>& shape) {
    int64_t meshDevices = 1;
    for (const MeshAxis& axis : axes) meshDevices *= axis.size;
    ReshardOracle oracle(axes, std::vector<int64_t>(shape.size(), meshDevices));
    constexpr int kBatch = 4096;
    std::vector<std::pair<int, int>> batch;
    int64_t numErrors = 0;
    std::vector<std::string> sampleErrors;
    auto flush = [&]() {
      std::vector<PassPlan> plans = runPassWithShape(
          context, oracle, shape, batch, /*checkPaddingInvariants=*/true);
      for (auto [pair, plan] : llvm::zip_equal(batch, plans)) {
        if (!plan.error.empty()) {
          ++numErrors;
          if (sampleErrors.size() < 10) {
            sampleErrors.push_back(
                plan.error + ": " + oracle.toString(oracle.state(pair.first)) +
                " -> " + oracle.toString(oracle.state(pair.second)));
          }
        }
      }
      batch.clear();
    };
    for (int src = 0; src < oracle.numStates(); ++src) {
      for (int tgt = 0; tgt < oracle.numStates(); ++tgt) {
        if (src != tgt) batch.push_back({src, tgt});
      }
      if (batch.size() >= kBatch) flush();
    }
    if (!batch.empty()) flush();
    EXPECT_EQ(numErrors, 0)
        << "First error: " << (sampleErrors.empty() ? "" : sampleErrors[0]);
  }

  std::string describe(const ReshardOracle& oracle, const Counterexample& c) {
    std::ostringstream os;
    os << oracle.toString(oracle.state(c.src)) << " -> "
       << oracle.toString(oracle.state(c.tgt)) << "\n  greedy ("
       << c.greedy.elems << " elems, " << c.greedy.ops << " ops):";
    for (const Step& step : c.greedySteps) {
      os << " " << moveName(step.kind) << " "
         << oracle.toString(oracle.state(step.state)) << ";";
    }
    os << "\n  oracle (" << c.oracle.elems << " elems, " << c.oracle.ops
       << " ops):";
    for (const Step& step : c.oracleSteps) {
      os << " " << moveName(step.kind) << " "
         << oracle.toString(oracle.state(step.state)) << ";";
    }
    return os.str();
  }

  void report(const ReshardOracle& oracle, const SweepStats& stats,
              std::vector<Counterexample>& counterexamples, int numToPrint) {
    std::ostringstream os;
    double excessPct =
        stats.totalOracleElems == 0
            ? 0.0
            : 100.0 * (stats.totalPassElems - stats.totalOracleElems) /
                  stats.totalOracleElems;
    os << "states=" << oracle.numStates() << " pairs=" << stats.pairs
       << " suboptimal_elems=" << stats.subOptimalElems << " ("
       << 100.0 * stats.subOptimalElems / std::max<int64_t>(1, stats.pairs)
       << "%) excess_elems_pct=" << excessPct
       << "% suboptimal_ops_only=" << stats.subOptimalOps
       << " max_ratio=" << stats.maxRatio << " errors=" << stats.numErrors
       << "\n";
    for (const std::string& error : stats.errors) {
      os << "  error: " << error << "\n";
    }
    for (const auto& [bucket, count] : stats.ratioHistogram) {
      os << "  ratio " << bucket << ": " << count << "\n";
    }
    // Smallest counterexamples first: fewest atoms, then largest gap.
    auto numAtoms = [&](const Counterexample& c) {
      int n = 0;
      for (int s : {c.src, c.tgt}) {
        for (const auto& dimAtoms : oracle.state(s)) n += dimAtoms.size();
      }
      return n;
    };
    std::stable_sort(
        counterexamples.begin(), counterexamples.end(),
        [&](const Counterexample& a, const Counterexample& b) {
          bool aElems = a.greedy.elems > a.oracle.elems;
          bool bElems = b.greedy.elems > b.oracle.elems;
          return std::make_tuple(!aElems, numAtoms(a),
                                 -(a.greedy.elems - a.oracle.elems)) <
                 std::make_tuple(!bElems, numAtoms(b),
                                 -(b.greedy.elems - b.oracle.elems));
        });
    // Most common (greedy -> oracle) collective sequences among
    // bytes-suboptimal pairs.
    auto signature = [](const std::vector<Step>& steps) {
      std::string result;
      for (const Step& step : steps) {
        if (!result.empty()) result += ",";
        result += moveName(step.kind);
      }
      return result;
    };
    std::map<std::string, int64_t> patterns;
    std::map<std::string, const Counterexample*> patternExample;
    for (const Counterexample& c : counterexamples) {
      if (c.greedy.elems > c.oracle.elems) {
        std::string pattern =
            signature(c.greedySteps) + " => " + signature(c.oracleSteps);
        ++patterns[pattern];
        patternExample.try_emplace(pattern, &c);
      }
    }
    std::vector<std::pair<int64_t, std::string>> sortedPatterns;
    for (const auto& [pattern, count] : patterns) {
      sortedPatterns.push_back({count, pattern});
    }
    std::sort(sortedPatterns.rbegin(), sortedPatterns.rend());
    for (int i = 0; i < std::min<int>(12, sortedPatterns.size()); ++i) {
      os << "  pattern x" << sortedPatterns[i].first << ": "
         << sortedPatterns[i].second << "\n    e.g. "
         << describe(oracle, *patternExample[sortedPatterns[i].second]) << "\n";
    }
    for (int i = 0; i < std::min<int>(numToPrint, counterexamples.size());
         ++i) {
      os << describe(oracle, counterexamples[i]) << "\n";
    }
    llvm::errs() << os.str();
  }

  MLIRContext context;
};

std::vector<int> allStates(const ReshardOracle& oracle) {
  std::vector<int> result(oracle.numStates());
  for (int i = 0; i < oracle.numStates(); ++i) result[i] = i;
  return result;
}

TEST_F(ReshardOracleTest, OracleMatchesHandComputedCosts) {
  ReshardOracle oracle({{"x", 2}, {"y", 4}}, {8, 16});
  std::vector<Step> parent;
  int replicated = *oracle.indexOf(State{{}, {}});
  std::vector<Cost> dist = oracle.shortestFrom(replicated, &parent);
  // Slicing is free.
  EXPECT_EQ(dist[*oracle.indexOf(State{{0}, {1, 2}})], (Cost{0, 1}));
  int xOnDim0 = *oracle.indexOf(State{{0}, {}});
  dist = oracle.shortestFrom(xOnDim0, &parent);
  // Gathering "x" receives the other half of the tensor (64 elements).
  EXPECT_EQ(dist[replicated], (Cost{64, 1}));
  // Moving "x" to dim 1 is one all-to-all sending half the local shard.
  EXPECT_EQ(dist[*oracle.indexOf(State{{}, {0}})], (Cost{32, 1}));
  // Replacing "x" with "y":(1)2 is one collective-permute.
  EXPECT_EQ(dist[*oracle.indexOf(State{{1}, {}})], (Cost{64, 1}));
  EXPECT_EQ(oracle.lowerBound(xOnDim0, *oracle.indexOf(State{{1}, {}})), 64);
  int xy1OnDim0 = *oracle.indexOf(State{{0, 1}, {}});
  dist = oracle.shortestFrom(xy1OnDim0, &parent);
  // Two all-to-alls ("y":(1)2 then "x") cost 16 + 16 = 32 < cp(32) + a2a(24).
  EXPECT_EQ(dist[*oracle.indexOf(State{{}, {1, 0}})], (Cost{32, 2}));
  // Moving ["x", "y":(1)2] to dim 1 then gathering "y":(1)2 costs 24 + 32 = 56.
  EXPECT_EQ(dist[*oracle.indexOf(State{{}, {0}})], (Cost{56, 2}));
}

TEST_F(ReshardOracleTest, NonDivisibleShapesPreservePaddingInvariants) {
  verifyNonDivisibleAllPairs({{"a", 2}, {"b", 3}}, {5, 9});
  verifyNonDivisibleAllPairs({{"a", 2}, {"b", 4}}, {6, 10});
}

struct SweepCase {
  std::string name;
  std::vector<MeshAxis> axes;
  std::vector<int64_t> shape;
  // Number of pairs where the pass communicates more than the oracle. Lower
  // this when the pass improves; it must never go up. Negative means unchecked.
  int64_t maxSubOptimalElems;
  // Number of pairs where the pass fails or emits a wrong sequence.
  int64_t maxErrors;
};

class ReshardOracleSweepTest : public ReshardOracleTest,
                               public ::testing::WithParamInterface<SweepCase> {
};

TEST_P(ReshardOracleSweepTest, AllPairs) {
  const SweepCase& c = GetParam();
  ReshardOracle oracle(c.axes, c.shape);
  std::vector<Counterexample> counterexamples;
  SweepStats stats = sweep(oracle, allStates(oracle), &counterexamples, c.name);
  ASSERT_FALSE(HasFatalFailure());
  llvm::errs() << "== " << c.name << "\n";
  report(oracle, stats, counterexamples, 8);
  if (c.maxErrors >= 0) {
    EXPECT_LE(stats.numErrors, c.maxErrors);
  }
  if (c.maxSubOptimalElems >= 0) {
    EXPECT_LE(stats.subOptimalElems, c.maxSubOptimalElems);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SmallMeshes, ReshardOracleSweepTest,
    ::testing::Values(
        SweepCase{"2x3_rank2", {{"a", 2}, {"b", 3}}, {6, 12}, 20, 0},
        SweepCase{"2x4_rank2", {{"a", 2}, {"b", 4}}, {8, 16}, 372, 0},
        SweepCase{"2x4_rank3", {{"a", 2}, {"b", 4}}, {8, 16, 24}, 2970, 0},
        SweepCase{
            "2x3x4_rank2", {{"a", 2}, {"b", 3}, {"c", 4}}, {24, 48}, 21406, 21},
        SweepCase{
            "2x2x4_rank2", {{"a", 2}, {"b", 2}, {"c", 4}}, {16, 32}, 8400, 0}),
    [](const ::testing::TestParamInfo<SweepCase>& info) {
      return info.param.name;
    });

std::vector<SweepCase> largeSweepCases() {
  const char* env = std::getenv("SDY_RUN_LARGE_ORACLE_SWEEP");
  if (!env || std::string_view(env) != "1") {
    return {};
  }
  return {
      SweepCase{"2x3x4_rank3",
                {{"a", 2}, {"b", 3}, {"c", 4}},
                {24, 48, 72},
                210609,
                138},
      SweepCase{"2x2x4_rank3",
                {{"a", 2}, {"b", 2}, {"c", 4}},
                {16, 32, 48},
                112896,
                0},
      SweepCase{"2x8_rank3", {{"a", 2}, {"b", 8}}, {16, 32, 48}, 112896, 0},
      SweepCase{"2x3x4x5_rank2",
                {{"a", 2}, {"b", 3}, {"c", 4}, {"d", 5}},
                {120, 240},
                1236323,
                1408},
  };
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ReshardOracleSweepTest);

INSTANTIATE_TEST_SUITE_P(LargeSweep, ReshardOracleSweepTest,
                         ::testing::ValuesIn(largeSweepCases()),
                         [](const ::testing::TestParamInfo<SweepCase>& info) {
                           return info.param.name;
                         });

}  // namespace
}  // namespace sdy
}  // namespace mlir
