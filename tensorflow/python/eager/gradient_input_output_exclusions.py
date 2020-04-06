# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Script to generate inputs/outputs exclusion lists for GradientTape.

To use this script:

bazel run tensorflow/python/eager:gradient_input_output_exclusions -- \
  $PWD/tensorflow/python/eager/pywrap_gradient_exclusions.cc
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import liveness
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops

_GENERATED_FILE_HEADER = """/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Inputs/Outputs exclusion lists for GradientTape.
//
// This file is MACHINE GENERATED! Do not edit.
// Generated by: tensorflow/python/eager/gen_gradient_input_output_exclusions.py
"""

_INCLUDES = """
#include "tensorflow/python/eager/pywrap_gradient_exclusions.h"

#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"

using tensorflow::string;

namespace {
// Keep static data in a format that's easy to init statically.
struct OpIndexInfo {
  const char *op_name;
  int num_indices;
  std::array<int, 4> unused_indices;
};

// Helper function to initialize FlatMap<string,FlatSet> from OpIndexInfo.
template <typename T>
auto OpGradientInfoInit(const T &a) {
  auto *m = new tensorflow::gtl::FlatMap<string, tensorflow::gtl::FlatSet<int>>;
  for (const auto &item : a) {
    m->emplace(string(item.op_name),
               tensorflow::gtl::FlatSet<int>(
                   item.unused_indices.begin(),
                   item.unused_indices.begin() + item.num_indices));
  }
  return m;
}
}  // namespace
"""

_EXCLUDED_OPS = [
    # Composite ops with custom gradient functions.
    "If",
    "StatelessIf",
    "While",
    "StatelessWhile",
    "Case",

    # TF Lite. These ops only appear in OSS.
    # TODO(srbs): Find a better way to filter these out.
    "AudioMicrofrontend",
]


class _SubscriptUseTracker(transformer.Base):
    """Track uses of composite names, excluding certain names when subscripted."""

    def __init__(self, ctx, exclude_when_subscripted):
        super(_SubscriptUseTracker, self).__init__(ctx)
        self.exclude = exclude_when_subscripted
        self.reads = set()
        self.complex_reads = set()

    def visit_Attribute(self, node):
        """Visits attribute nodes in the AST."""
        if anno.hasanno(node, anno.Basic.QN):
            qn = anno.getanno(node, anno.Basic.QN)
            if isinstance(node.ctx, gast.Load):
                self.reads.add(qn)
        node = self.generic_visit(node)
        return node

    def visit_Subscript(self, node):
        """Visits nodes with subscript in the AST."""
        if anno.hasanno(node, anno.Basic.QN):
            qn = anno.getanno(node, anno.Basic.QN)
            if isinstance(node.ctx, gast.Load):
                self.reads.add(qn)
        elif not isinstance(node.slice, gast.Index):
            if anno.hasanno(node, anno.Basic.QN):
                self.complex_reads.add(anno.getanno(node, anno.Basic.QN))
            elif anno.hasanno(node.value, anno.Basic.QN):
                self.complex_reads.add(anno.getanno(node.value, anno.Basic.QN))
        value_qn = anno.getanno(node.value, anno.Basic.QN, None)
        if value_qn in self.exclude:
            node.value = self.generic_visit(node.value)
        else:
            node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)
        return node


class _FunctionCallsTracker(transformer.Base):
    """Tracks any function calls made with a given first argument name."""

    def __init__(self, ctx, first_argument_name):
        super(_FunctionCallsTracker, self).__init__(ctx)
        self.first_argument_name = first_argument_name
        self.calls = set()

    def visit_Name(self, node):
        node = self.generic_visit(node)
        if isinstance(node.ctx, gast.Load) and node.id in self.ctx.info.namespace:
            anno.setanno(node, "static_value",
                         self.ctx.info.namespace[node.id])
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        parent_val = anno.getanno(node.value, "static_value", default=None)
        if parent_val is not None:
            if hasattr(parent_val, node.attr):
                anno.setanno(node, "static_value",
                             getattr(parent_val, node.attr))
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (node.args and anno.getanno(node.args[0], anno.Basic.QN,
                                       None) == self.first_argument_name):
            fn_object = anno.getanno(node.func, "static_value", None)
            if fn_object is not None:
                self.calls.add(fn_object)
        return node


_ALL = object()


def _live_tensors(f, attr_name="inputs"):
    """Returns the indices of the used inputs.

    Note: This currently only handles direct index accesses e.g. op.inputs[1].
    If the function has slicing or list comprehension on attr_name then returns
    _ALL. This ensure that this is correct even if inefficient.

    Args:
      f: A grad function, taking the op as first argument.
      attr_name: op attr to track. "inputs" or "outputs".

    Returns:
      Either one of:
        * set of integers representing individual indices of inputs used
        * the value _ALL, if indices are used but cannot be determined which
        * empty set, if no inputs are used
    """
    node, _ = parser.parse_entity(f, ())
    entity_info = transformer.EntityInfo(
        name=f.__name__,
        source_code=None,
        source_file=None,
        future_features=(),
        namespace=sys.modules[f.__module__].__dict__)
    ctx = transformer.Context(entity_info, None, None)

    graphs = cfg.build(node)
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx, None)
    node = liveness.resolve(node, ctx, graphs)

    op_arg_name = anno.getanno(node.args.args[0], anno.Basic.QN)
    op_inputs_outputs_name = qual_names.QN(op_arg_name, attr=attr_name)

    special_tracker = _SubscriptUseTracker(ctx, (op_inputs_outputs_name,))
    node = special_tracker.visit(node)

    live_vars_in = anno.getanno(node.body[0], anno.Static.LIVE_VARS_IN)
    inputs_outputs_used_qns = set()
    for v in special_tracker.complex_reads:
        # Complicated patterns like op.inputs[:3]. Could be smarter about them
        # if they matter much.
        if v == op_inputs_outputs_name:
            return _ALL
    for v in live_vars_in:
        if v in special_tracker.reads:
            if (v.has_subscript() and v.parent == op_inputs_outputs_name):
                inputs_outputs_used_qns.add(v)
            elif v == op_inputs_outputs_name:
                # When op.{attr_name} is used directly, assume all tensors are
                # used for now. In that case, no point digging further.
                # TODO(mdan): We can descend into tuple expansions.
                return _ALL

    function_calls_tracker = _FunctionCallsTracker(ctx, op_arg_name)
    node = function_calls_tracker.visit(node)

    input_output_indices = set()

    for called_f in function_calls_tracker.calls:
        child_indices = _live_tensors(called_f, attr_name=attr_name)
        if child_indices is _ALL:
            return _ALL
        input_output_indices |= child_indices

    for v in inputs_outputs_used_qns:
        assert v.has_subscript()
        _, subscript = v.qn
        if not subscript.is_simple():
            # Not a number, assuming it can be anything.
            return _ALL
        subscript_val, = subscript.qn
        if not isinstance(subscript_val, qual_names.NumberLiteral):
            # Not a number, assuming it can be anything.
            return _ALL
        input_output_indices.add(subscript_val.value)
    return input_output_indices


def _get_num_inputs_outputs(op_type):
    """Returns (num_inputs, num_outputs).

    Args:
      op_type: String. The type of the Operation. Used to lookup the op in the
        registry.

    Returns:
      (num_inputs, num_outputs), for either num_inputs or num_outputs if the value
      can't be statically inferred from the OpDef alone or of the OpDef lookup
      fails, -1 is returned.
    """

    def _is_list_arg(arg):
        return arg.number_attr or arg.type_list_attr

    def _count_args(arg_defs):
        for arg in arg_defs:
            if _is_list_arg(arg):
                # Op has list type args which could be variable.
                return -1
        return len(arg_defs)

    op_def = op_def_registry.get(op_type)
    if not op_def:
        return -1, -1
    return _count_args(op_def.input_arg), _count_args(op_def.output_arg)


def get_entries(attr_name):
    """Returns the dict of entries.

    Each entry is of the form {op_name, {true|false, indices}}

    true: All values are unused.
    false: `indices` are the only unused indices.

    Note: ops for which all values are used are not printed.

    Args:
      attr_name: inputs or outputs.

    Returns:
      A dict from op_type to formatted entry in the dict.
    """
    assert attr_name in ["inputs", "outputs"]
    entries = {}
    for op_type in ops._gradient_registry.list():  # pylint: disable=protected-access
        if op_type in _EXCLUDED_OPS:
            continue
        num_values = _get_num_inputs_outputs(op_type)[0 if attr_name ==
                                                      "inputs" else 1]
        gradient_fn = ops._gradient_registry.lookup(
            op_type)  # pylint: disable=protected-access
        if gradient_fn is None:
            # NotDifferentiable
            if num_values != -1:
                entries[op_type] = "{\"%s\"}," % op_type
            continue
        used_tensors = _live_tensors(gradient_fn, attr_name=attr_name)
        if used_tensors is _ALL:
            continue
        elif not used_tensors:
            entries[op_type] = "{\"%s\"}," % op_type
        else:
            all_tensors = set(range(num_values))
            unused_tensors = all_tensors - used_tensors
            if unused_tensors:
                unused_tensor_list = sorted(list(unused_tensors))
                entries[op_type] = "{\"%s\", %d, {%s}}," % (
                    op_type, len(unused_tensor_list), ", ".join(
                        str(i) for i in unused_tensor_list))
    return entries


def get_function(name, entries):
    """Generates lookup function with given name and lookup table entries."""
    contents = """
absl::optional<tensorflow::gtl::FlatSet<int>> {name}(
    const tensorflow::string &op_name) {{
  static std::array<OpIndexInfo, {count}> a = {{{{
""".format(
        name=name, count=len(entries) + 1)
    contents += "      "
    contents += "\n      ".join(entries[op_type]
                                for op_type in sorted(entries))
    contents += "\n      {\"VarHandleOp\"},"
    contents += """
  }};
  static const auto &m = *OpGradientInfoInit(a);

  auto it = m.find(op_name);
  if (it != m.end()) {
    return it->second;
  }
  return absl::nullopt;
}
"""
    return contents


def get_contents():
    """Returns contents for the generated file."""
    contents = ""
    contents += _GENERATED_FILE_HEADER + _INCLUDES
    contents += get_function("OpGradientUnusedInputIndices",
                             get_entries("inputs"))
    contents += get_function("OpGradientUnusedOutputIndices",
                             get_entries("outputs"))
    return contents


def main(output_file):
    with open(output_file, "w") as fp:
        fp.write(get_contents())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("output", metavar="O",
                            type=str, help="Output file.")
    args = arg_parser.parse_args()
    main(args.output)
