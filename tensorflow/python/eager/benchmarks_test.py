# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
r"""Benchmarks for low-level eager execution primitives.

To run CPU benchmarks:
  bazel run -c opt benchmarks_test -- --benchmarks=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" benchmarks_test -- \
    --benchmarks=.

To run a subset of benchmarks using --benchmarks flag.
--benchmarks: the list of benchmarks to run. The specified value is interpreted
as a regular expression and any benchmark whose name contains a partial match
to the regular expression is executed.
e.g. --benchmarks=".*matmul*." will run all matmul related benchmarks.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import def_function
from tensorflow.python.eager import forwardprop
from tensorflow.python.eager import function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops


CPU = "/device:CPU:0"
GPU = "/device:GPU:0"
GLOBAL_TEST_VALUE = None


def c_tfe_py_fastpath_execute(a, b, transpose_a=False, transpose_b=False, name=None):
    ctx = context.context()
    assert (
        ctx.executing_eagerly()
    ), "The prototype doesn't contain C code for graph construction"
    try:
        return pywrap_tfe.TFE_Py_FastPathExecute(
            ctx._handle,
            ctx.device_name,
            "MatMul",
            name,
            ctx.op_callbacks,
            a,
            b,
            "transpose_a",
            transpose_a,
            "transpose_b",
            transpose_b,
        )
    except core._NotOkStatusException as e:
        if name is not None:
            message = e.message + " name: " + name
        else:
            message = e.message
        six.raise_from(core._status_to_exception(e.code, message), None)


def run_benchmark(func, num_iters, execution_mode=None):
    ctx = context.context()
    with context.execution_mode(execution_mode):
        # call func to warm up
        func()
        if execution_mode == context.ASYNC:
            ctx.executor.wait()
        start = time.time()
        for _ in xrange(num_iters):
            func()
        if execution_mode == context.ASYNC:
            ctx.executor.wait()
        end = time.time()

        return end - start


class MicroBenchmarks(test.Benchmark):
    def __init__(self):
        # TODO(b/153054118): Add tf.RandomUniform
        if not context.is_tfrt_enabled():
            # used for multiply benchmarks
            self._m_2 = random_ops.random_uniform([2])

            # used for matmul benchmarks
            self._m_2_by_2 = random_ops.random_uniform((2, 2))
            self._m_100_by_784 = random_ops.random_uniform((100, 784))

        self._num_iters_2_by_2 = 30000
        self._num_iters_100_by_784 = 30000

    def _run(self, func, num_iters, execution_mode=None):
        total_time = run_benchmark(func, num_iters, execution_mode)
        mean_us = total_time * 1e6 / num_iters
        self.report_benchmark(
            iters=num_iters,
            wall_time=mean_us,
            extras={
                "examples_per_sec": float("{0:.3f}".format(num_iters / total_time)),
                "us_per_example": float("{0:.3f}".format(total_time * 1e6 / num_iters)),
            },
        )

    def benchmark_create_np_array(self):
        def func():
            return np.array([3.0])

        self._run(func, 30000)

    def _benchmark_create_tensor(self, value, dtype, device):
        """Benchmark overheads of creating a Tensor object."""
        ctx = context.context()
        if device == GPU:
            # Warmup the GPU
            ops.EagerTensor(value, device=device)

        def func():
            ops.EagerTensor(value, device=device, dtype=dtype)

        self._run(func, 30000)

    def _benchmark_create_constant(self, value, dtype, cached=True):
        global GLOBAL_TEST_VALUE
        GLOBAL_TEST_VALUE = value

        def cached_func():
            constant_op.constant(value, dtype=dtype)

        def uncached_func():
            global GLOBAL_TEST_VALUE
            GLOBAL_TEST_VALUE += 1
            constant_op.constant(GLOBAL_TEST_VALUE, dtype=dtype)

        func = cached_func if cached else uncached_func

        with ops.device("GPU:0" if context.num_gpus() else "CPU:0"):
            for _ in range(1000):
                func()  # Warmup.
            self._run(func, 3000)

    def benchmark_create_float_constant(self):
        self._benchmark_create_constant(42.0, dtype=None)

    def benchmark_create_float_constant_uncached(self):
        self._benchmark_create_constant(42.0, dtype=None, cached=False)

    def benchmark_create_int32_constant(self):
        if context.num_gpus():
            return  # int32 constants are always allocated on CPU.

        self._benchmark_create_constant(42, dtype=dtypes.int32)

    def benchmark_create_int32_constant_uncached(self):
        if context.num_gpus():
            return  # int32 constants are always allocated on CPU.

        self._benchmark_create_constant(42, dtype=dtypes.int32, cached=False)

    def _benchmark_add(self, a, b):
        def func():
            return memoryview(math_ops.add_v2(a, b))

        with ops.device("GPU:0" if context.num_gpus() else "CPU:0"):
            for _ in range(1000):
                func()  # Warmup.
            self._run(func, 30000)

    def benchmark_add_float_scalars(self):
        self._benchmark_add(42.0, 24.0)

    def benchmark_add_int32_scalars(self):
        self._benchmark_add(42, 24)

    def benchmark_add_float_scalar_tensor(self):
        tensor_a = constant_op.constant(42.0)
        tensor_b = constant_op.constant(24.0)
        self._benchmark_add(tensor_a, tensor_b)

    def benchmark_add_int32_scalar_tensor(self):
        tensor_a = constant_op.constant(42)
        tensor_b = constant_op.constant(24)
        self._benchmark_add(tensor_a, tensor_b)

    def benchmark_add_float_dense_tensor(self):
        tensor_a = constant_op.constant([[42.0, 42.0], [42.0, 42.0]])
        tensor_b = constant_op.constant([[24.0, 24.0], [24.0, 24.0]])
        self._benchmark_add(tensor_a, tensor_b)

    def benchmark_add_int32_dense_tensor(self):
        tensor_a = constant_op.constant([[42, 42], [42, 42]])
        tensor_b = constant_op.constant([[24, 24], [24, 24]])
        self._benchmark_add(tensor_a, tensor_b)

    def benchmark_create_float_tensor_from_list_CPU(self):
        self._benchmark_create_tensor([[3.0]], dtypes.float32.as_datatype_enum, CPU)

    def benchmark_create_float_tensor_from_np_array_CPU(self):
        self._benchmark_create_tensor(
            np.array([[3.0]], dtype=np.float32), dtypes.float32.as_datatype_enum, CPU
        )

    def benchmark_create_int32_tensor_from_list_CPU(self):
        self._benchmark_create_tensor([[3]], dtypes.int32.as_datatype_enum, CPU)

    def benchmark_create_int32_tensor_from_np_array_CPU(self):
        self._benchmark_create_tensor(
            np.array([[3]], dtype=np.int32), dtypes.int32.as_datatype_enum, CPU
        )

    def benchmark_create_float_tensor_from_list_GPU(self):
        if not context.num_gpus():
            return
        self._benchmark_create_tensor([[3.0]], dtypes.float32.as_datatype_enum, GPU)

    def benchmark_create_float_tensor_from_np_array_GPU(self):
        if not context.num_gpus():
            return
        self._benchmark_create_tensor(
            np.array([[3.0]], dtype=np.float32), dtypes.float32.as_datatype_enum, GPU
        )

    def benchmark_create_int32_tensor_from_list_GPU(self):
        # int32's are kept on host memory even when executing on GPU.
        if not context.num_gpus():
            return
        self._benchmark_create_tensor([[3]], dtypes.int32.as_datatype_enum, GPU)

    def benchmark_create_int32_tensor_from_np_array_GPU(self):
        # int32's are kept on host memory even when executing on GPU.
        if not context.num_gpus():
            return
        self._benchmark_create_tensor(
            np.array([[3]], dtype=np.int32), dtypes.int32.as_datatype_enum, GPU
        )

    def benchmark_index_tensor_with_literal(self):
        def func():
            return constant_op.constant([3.0])[0]

        self._run(func, 30000)

    def benchmark_index_tensor_with_tensor(self):
        def func(idx=constant_op.constant(0)):
            return constant_op.constant([3.0])[idx]

        self._run(func, 30000)

    def benchmark_index_tensor_with_np_array(self):
        def func(idx=np.array(0)):
            return constant_op.constant([3.0])[idx]

        self._run(func, 30000)

    def _benchmark_np_multiply(self, m, num_iters):
        a = m.cpu().numpy()

        def func():
            return a * a

        self._run(func, num_iters)

    def _benchmark_tf_multiply(self, m, num_iters):
        def func():
            return m * m

        self._run(func, num_iters)

    def _benchmark_tf_multiply_op(self, m, num_iters):
        def func():
            return math_ops.multiply(m, m)

        self._run(func, num_iters)

    def benchmark_np_multiply(self):
        self._benchmark_np_multiply(self._m_2, 30000)

    def benchmark_tf_multiply_CPU(self):
        with context.device(CPU):
            m = self._m_2.cpu()
            self._benchmark_tf_multiply(m, 30000)

    def benchmark_tf_multiply_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_2.gpu()
            self._benchmark_tf_multiply(m, 30000)

    def benchmark_tf_multiply_op_CPU(self):
        with context.device(CPU):
            m = self._m_2.cpu()
            self._benchmark_tf_multiply_op(m, 30000)

    def benchmark_tf_multiply_op_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_2.gpu()
            self._benchmark_tf_multiply_op(m, 30000)

    def benchmark_tf_identity(self):
        m = self._m_2
        self._run(lambda: gen_array_ops.identity(m), 30000)

    def benchmark_slowpath_tf_identity(self):
        self._run(lambda: gen_array_ops.identity(1), 30000)

    def benchmark_tfe_py_execute_identity(self):
        m = self._m_2
        ctx_handle = context.context()._handle
        attrs = ("T", self._m_2.dtype.as_datatype_enum)
        inputs = [m]

        def f():
            pywrap_tfe.TFE_Py_Execute(ctx_handle, None, "Identity", inputs, attrs, 1)

        self._run(f, 30000)

    def benchmark_tf_gradient_function_identity(self):
        with context.device(CPU):
            m = gen_array_ops.identity(self._m_2)
            self._run(
                lambda: backprop.gradients_function(gen_array_ops.identity, [0])(m),
                30000,
            )

    def benchmark_tf_gradient_forward_identity(self):
        with backprop.GradientTape() as tape:
            m = self._m_2
            tape.watch(m)
            self._run(lambda: gen_array_ops.identity(m), 30000)

    def benchmark_tf_gradient_tape_push_pop(self):
        def f():
            with backprop.GradientTape():
                pass

        self._run(f, 30000)

    def benchmark_tf_gradient_function_no_op(self):
        with context.device(CPU):
            m = gen_array_ops.identity(self._m_2)
            self._run(lambda: backprop.gradients_function(lambda x: x, [0])(m), 30000)

    def _benchmark_np_matmul(self, m, transpose_b, num_iters):
        a = m.cpu().numpy()
        b = a.T if transpose_b else a

        def func():
            return np.dot(a, b)

        self._run(func, num_iters)

    def _benchmark_tf_matmul(self, m, transpose_b, num_iters, execution_mode=None):
        def func():
            return math_ops.matmul(m, m, transpose_b=transpose_b)

        self._run(func, num_iters, execution_mode=execution_mode)

    def _benchmark_gen_math_ops_matmul(self, m, transpose_b, num_iters):
        def func():
            gen_math_ops.mat_mul(m, m, transpose_b=transpose_b)

        self._run(func, num_iters)

    def _benchmark_tfe_py_fastpath_execute_matmul(self, m, transpose_b, num_iters):
        def func():
            c_tfe_py_fastpath_execute(m, m, transpose_b=transpose_b)

        self._run(func, num_iters)

    def _benchmark_tfe_py_execute_matmul(self, m, transpose_b, num_iters):
        inputs = [m, m]
        # pylint: disable=protected-access
        ctx_handle = context.context()._handle
        # pylint: enable=protected-access
        device = context.context().device_name
        attrs = (
            "transpose_a",
            False,
            "transpose_b",
            transpose_b,
            "T",
            m.dtype.as_datatype_enum,
        )

        def func():
            pywrap_tfe.TFE_Py_Execute(ctx_handle, device, "MatMul", inputs, attrs, 1)

        self._run(func, num_iters)

    def _benchmark_defun_matmul(self, m, transpose_b, num_iters, execution_mode=None):
        f = function.defun(math_ops.matmul)

        def func():
            return f(m, m, transpose_b=transpose_b)

        self._run(func, num_iters, execution_mode=execution_mode)

    def _benchmark_nested_defun_matmul(self, m, transpose_b, num_iters):
        inner = function.defun(math_ops.matmul)

        @function.defun
        def outer(a, b, c, transpose_b):
            return math_ops.matmul(inner(a, b, transpose_b=transpose_b), c)

        def func():
            return outer(m, m, m, transpose_b=transpose_b)

        # Warmup before benchmark
        for _ in range(1000):
            func()
        self._run(func, num_iters)

    def _benchmark_defun_matmul_forward_backward(
        self, m, transpose_b, num_iters, execution_mode=None
    ):
        f = function.defun(math_ops.matmul)

        def func():
            with backprop.GradientTape() as gt:
                gt.watch(m)
                y = f(m, m, transpose_b=transpose_b)
            _ = gt.gradient(y, m)

        self._run(func, num_iters, execution_mode=execution_mode)

    def _benchmark_read_variable(self, m, num_iters):
        self._run(m.value, num_iters)

    def _benchmark_matmul_read_variable(self, m, num_iters):
        self._benchmark_gen_math_ops_matmul(m, transpose_b=False, num_iters=num_iters)

    def _benchmark_matmul_read_variable_with_tape(self, m, num_iters):
        with backprop.GradientTape() as tape:
            tape.watch(m)
            self._benchmark_gen_math_ops_matmul(
                m, transpose_b=False, num_iters=num_iters
            )

    def _benchmark_read_variable_with_tape(self, m, num_iters):
        with backprop.GradientTape() as tape:
            tape.watch(m)
            self._run(m.value, num_iters)

    # Benchmarks for A^2, A of dimension 2 by 2.
    def benchmark_np_matmul_2_by_2(self):
        self._benchmark_np_matmul(
            self._m_2_by_2, transpose_b=False, num_iters=self._num_iters_2_by_2
        )

    def benchmark_tf_matmul_2_by_2_CPU(self):
        with context.device(CPU):
            m = self._m_2_by_2.cpu()
            self._benchmark_tf_matmul(
                m, transpose_b=False, num_iters=self._num_iters_2_by_2
            )

    def benchmark_tf_matmul_2_by_2_CPU_async(self):
        with context.device(CPU):
            m = self._m_2_by_2.cpu()
            self._benchmark_tf_matmul(
                m,
                transpose_b=False,
                num_iters=self._num_iters_2_by_2,
                execution_mode=context.ASYNC,
            )

    def benchmark_gen_math_ops_matmul_2_by_2_CPU(self):
        with context.device(CPU):
            m = self._m_2_by_2.cpu()
            self._benchmark_gen_math_ops_matmul(
                m, transpose_b=False, num_iters=self._num_iters_2_by_2
            )

    def benchmark_tfe_py_fastpath_execute_matmul_2_by_2_CPU(self):
        with context.device(CPU):
            m = self._m_2_by_2.cpu()
            self._benchmark_tfe_py_fastpath_execute_matmul(
                m, transpose_b=False, num_iters=self._num_iters_2_by_2
            )

    def benchmark_tfe_py_execute_matmul_2_by_2_CPU(self):
        with context.device(CPU):
            m = self._m_2_by_2.cpu()
            self._benchmark_tfe_py_execute_matmul(
                m, transpose_b=False, num_iters=self._num_iters_2_by_2
            )

    def benchmark_defun_matmul_2_by_2_CPU(self):
        with context.device(CPU):
            m = self._m_2_by_2.cpu()
            self._benchmark_defun_matmul(
                m, transpose_b=False, num_iters=self._num_iters_2_by_2
            )

    def benchmark_defun_matmul_2_by_2_CPU_async(self):
        with context.device(CPU):
            m = self._m_2_by_2.cpu()
            self._benchmark_defun_matmul(
                m,
                transpose_b=False,
                num_iters=self._num_iters_2_by_2,
                execution_mode=context.ASYNC,
            )

    def benchmark_defun_matmul_forward_backward_2_by_2_CPU(self):
        with context.device(CPU):
            m = self._m_2_by_2.cpu()
            self._benchmark_defun_matmul_forward_backward(
                m, transpose_b=False, num_iters=self._num_iters_2_by_2
            )

    def benchmark_defun_matmul_forward_backward_2_by_2_CPU_async(self):
        with context.device(CPU):
            m = self._m_2_by_2.cpu()
            self._benchmark_defun_matmul_forward_backward(
                m,
                transpose_b=False,
                num_iters=self._num_iters_2_by_2,
                execution_mode=context.ASYNC,
            )

    def benchmark_tf_matmul_2_by_2_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_2_by_2.gpu()
            self._benchmark_tf_matmul(
                m, transpose_b=False, num_iters=self._num_iters_2_by_2
            )

    def benchmark_tf_matmul_2_by_2_GPU_async(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_2_by_2.gpu()
            self._benchmark_tf_matmul(
                m,
                transpose_b=False,
                num_iters=self._num_iters_2_by_2,
                execution_mode=context.ASYNC,
            )

    def benchmark_gen_math_ops_matmul_2_by_2_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_2_by_2.gpu()
            self._benchmark_gen_math_ops_matmul(
                m, transpose_b=False, num_iters=self._num_iters_2_by_2
            )

    def benchmark_tfe_py_execute_matmul_2_by_2_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_2_by_2.gpu()
            self._benchmark_tfe_py_execute_matmul(
                m, transpose_b=False, num_iters=self._num_iters_2_by_2
            )

    def benchmark_defun_matmul_2_by_2_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_2_by_2.gpu()
            self._benchmark_defun_matmul(
                m, transpose_b=False, num_iters=self._num_iters_2_by_2
            )

    def benchmark_defun_matmul_2_by_2_GPU_async(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_2_by_2.gpu()
            self._benchmark_defun_matmul(
                m,
                transpose_b=False,
                num_iters=self._num_iters_2_by_2,
                execution_mode=context.ASYNC,
            )

    def benchmark_nested_defun_matmul_2_by_2(self):
        m = self._m_2_by_2.cpu()
        self._benchmark_nested_defun_matmul(
            m, transpose_b=False, num_iters=self._num_iters_2_by_2
        )

    # Benchmarks for AA.T, A of dimension 100 by 784.
    def benchmark_np_matmul_100_by_784(self):
        self._benchmark_np_matmul(
            self._m_100_by_784, transpose_b=True, num_iters=self._num_iters_100_by_784
        )

    def benchmark_tf_matmul_100_by_784_CPU(self):
        with context.device(CPU):
            m = self._m_100_by_784.cpu()
            self._benchmark_tf_matmul(
                m, transpose_b=True, num_iters=self._num_iters_100_by_784
            )

    def benchmark_tf_matmul_100_by_784_CPU_async(self):
        with context.device(CPU):
            m = self._m_100_by_784.cpu()
            self._benchmark_tf_matmul(
                m,
                transpose_b=True,
                num_iters=self._num_iters_100_by_784,
                execution_mode=context.ASYNC,
            )

    def benchmark_gen_math_ops_matmul_100_by_784_CPU(self):
        with context.device(CPU):
            m = self._m_100_by_784.cpu()
            self._benchmark_gen_math_ops_matmul(
                m, transpose_b=True, num_iters=self._num_iters_100_by_784
            )

    def benchmark_tfe_py_fastpath_execute_matmul_100_by_784_CPU(self):
        with context.device(CPU):
            m = self._m_100_by_784.cpu()
            self._benchmark_tfe_py_fastpath_execute_matmul(
                m, transpose_b=True, num_iters=self._num_iters_100_by_784
            )

    def benchmark_tfe_py_execute_matmul_100_by_784_CPU(self):
        with context.device(CPU):
            m = self._m_100_by_784.cpu()
            self._benchmark_tfe_py_execute_matmul(
                m, transpose_b=True, num_iters=self._num_iters_100_by_784
            )

    def benchmark_defun_matmul_100_by_784_CPU(self):
        with context.device(CPU):
            m = self._m_100_by_784.cpu()
            self._benchmark_defun_matmul(
                m, transpose_b=True, num_iters=self._num_iters_100_by_784
            )

    def benchmark_tf_matmul_100_by_784_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_100_by_784.gpu()
            self._benchmark_tf_matmul(
                m, transpose_b=True, num_iters=self._num_iters_100_by_784
            )

    def benchmark_tf_matmul_100_by_784_GPU_async(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_100_by_784.gpu()
            self._benchmark_tf_matmul(
                m,
                transpose_b=True,
                num_iters=self._num_iters_100_by_784,
                execution_mode=context.ASYNC,
            )

    def benchmark_gen_math_ops_matmul_100_by_784_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_100_by_784.gpu()
            self._benchmark_gen_math_ops_matmul(
                m, transpose_b=True, num_iters=self._num_iters_100_by_784
            )

    def benchmark_tfe_py_execute_matmul_100_by_784_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_100_by_784.gpu()
            self._benchmark_tfe_py_execute_matmul(
                m, transpose_b=True, num_iters=self._num_iters_100_by_784
            )

    def benchmark_defun_matmul_100_by_784_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = self._m_100_by_784.gpu()
            self._benchmark_defun_matmul(
                m, transpose_b=True, num_iters=self._num_iters_100_by_784
            )

    def benchmark_nested_defun_matmul_100_by_784(self):
        m = self._m_100_by_784.gpu()
        self._benchmark_nested_defun_matmul(
            m, transpose_b=True, num_iters=self._num_iters_100_by_784
        )

    def _benchmark_forwardprop_matmul_CPU(self, shape):
        with ops.device(CPU):
            m = random_ops.random_uniform(shape).cpu()
            tangent = random_ops.random_uniform(shape).cpu()

            def func():
                with forwardprop.ForwardAccumulator(m, tangent) as acc:
                    result = math_ops.matmul(m, m, transpose_b=True)
                return result, acc.jvp(result)

            # Warmup before benchmark
            for _ in range(100):
                func()
            self._run(func, 3000)

    def _benchmark_forwardprop_in_defun_matmul_CPU(self, shape):
        with ops.device(CPU):

            @def_function.function
            def compiled_function(x, tangent):
                with forwardprop.ForwardAccumulator(x, tangent) as acc:
                    result = math_ops.matmul(x, x, transpose_b=True)
                return result, acc.jvp(result)

            m = random_ops.random_uniform(shape).cpu()
            tangent = random_ops.random_uniform(shape).cpu()

            def func():
                return compiled_function(m, tangent)

            # Warmup before benchmark
            for _ in range(100):
                func()
            self._run(func, 3000)

    def _benchmark_forwardprop_in_defun_of_defun_matmul_CPU(self, shape):
        with ops.device(CPU):
            matmul = def_function.function(math_ops.matmul)

            @def_function.function()
            def compiled_function(x, tangent):
                with forwardprop.ForwardAccumulator(x, tangent) as acc:
                    result = matmul(x, x, transpose_b=True)
                return result, acc.jvp(result)

            m = random_ops.random_uniform(shape).cpu()
            tangent = random_ops.random_uniform(shape).cpu()

            def func():
                return compiled_function(m, tangent)

            # Warmup before benchmark
            for _ in range(100):
                func()
            self._run(func, 3000)

    def _benchmark_forwardprop_of_defun_matmul_CPU(self, shape):
        with ops.device(CPU):
            m = random_ops.random_uniform(shape).cpu()
            tangent = random_ops.random_uniform(shape).cpu()
            matmul = def_function.function(math_ops.matmul)

            def func():
                with forwardprop.ForwardAccumulator(m, tangent) as acc:
                    result = matmul(m, m, transpose_b=True)
                return result, acc.jvp(result)

            # Warmup before benchmark
            for _ in range(100):
                func()
            self._run(func, 3000)

    def benchmark_forwardprop_matmul_256_by_2096_CPU(self):
        self._benchmark_forwardprop_matmul_CPU(shape=(256, 2096))

    def benchmark_forwardprop_in_defun_matmul_256_by_2096_CPU(self):
        self._benchmark_forwardprop_in_defun_matmul_CPU(shape=(256, 2096))

    def benchmark_forwardprop_in_defun_of_defun_matmul_256_by_2096_CPU(self):
        self._benchmark_forwardprop_in_defun_of_defun_matmul_CPU(shape=(256, 2096))

    def benchmark_forwardprop_of_defun_matmul_256_by_2096_CPU(self):
        self._benchmark_forwardprop_of_defun_matmul_CPU(shape=(256, 2096))

    def benchmark_forwardprop_matmul_100_by_784_CPU(self):
        self._benchmark_forwardprop_matmul_CPU(shape=(100, 784))

    def benchmark_forwardprop_in_defun_matmul_100_by_784_CPU(self):
        self._benchmark_forwardprop_in_defun_matmul_CPU(shape=(100, 784))

    def benchmark_forwardprop_in_defun_of_defun_matmul_100_by_784_CPU(self):
        self._benchmark_forwardprop_in_defun_of_defun_matmul_CPU(shape=(100, 784))

    def benchmark_forwardprop_of_defun_matmul_100_by_784_CPU(self):
        self._benchmark_forwardprop_of_defun_matmul_CPU(shape=(100, 784))

    def _benchmark_tf_reduce_logsumexp(
        self, device=CPU, execution_mode=None, defunc=False
    ):
        with context.device(device):
            x = constant_op.constant([[1, 0.0], [0.0, 0.0]])
            if defunc:
                reduce_func = def_function.function(math_ops.reduce_logsumexp)

                def func():
                    return reduce_func(x)

            else:

                def func():
                    return math_ops.reduce_logsumexp(x)

            self._run(func, 3000, execution_mode=execution_mode)

    def benchmark_tf_reduce_logsumexp_CPU(self):
        self._benchmark_tf_reduce_logsumexp()

    def benchmark_tf_reduce_logsumexp_CPU_async(self):
        self._benchmark_tf_reduce_logsumexp(execution_mode=context.ASYNC)

    def benchmark_tf_reduce_logsumexp_GPU(self):
        self._benchmark_tf_reduce_logsumexp(device=GPU)

    def benchmark_tf_reduce_logsumexp_GPU_async(self):
        self._benchmark_tf_reduce_logsumexp(device=GPU, execution_mode=context.ASYNC)

    def benchmark_tf_reduce_logsumexp_CPU_defunc(self):
        self._benchmark_tf_reduce_logsumexp(defunc=True)

    def benchmark_tf_reduce_logsumexp_CPU_async_defun(self):
        self._benchmark_tf_reduce_logsumexp(execution_mode=context.ASYNC, defunc=True)

    def benchmark_tf_reduce_logsumexp_GPU_defun(self):
        self._benchmark_tf_reduce_logsumexp(device=GPU, defunc=True)

    def benchmark_tf_reduce_logsumexp_GPU_async_defun(self):
        self._benchmark_tf_reduce_logsumexp(
            device=GPU, execution_mode=context.ASYNC, defunc=True
        )

    def _benchmark_tf_tensordot(self, device=CPU, execution_mode=None):
        with context.device(device):
            a = array_ops.ones((2, 2))
            b = array_ops.ones((2, 2))

            def func():
                return math_ops.tensordot(a, b, [[1], [0]])

            self._run(func, 30000, execution_mode=execution_mode)

    def benchmark_tf_tensordot_CPU(self):
        self._benchmark_tf_tensordot()

    def benchmark_tf_tensordot_CPU_async(self):
        self._benchmark_tf_tensordot(execution_mode=context.ASYNC)

    def benchmark_tf_tensordot_GPU(self):
        self._benchmark_tf_tensordot(device=GPU)

    def benchmark_tf_tensordot_GPU_async(self):
        self._benchmark_tf_tensordot(device=GPU, execution_mode=context.ASYNC)

    def _benchmark_tf_zeros(self, shape, dtype, device=CPU):
        with context.device(device):

            def func():
                return array_ops.zeros(shape, dtype)

            self._run(func, 3000)

    def benchmark_tf_zeros_2_by_2_float32_CPU(self):
        self._benchmark_tf_zeros((2, 2), dtypes.float32)

    def benchmark_tf_zeros_2_by_2_bool_CPU(self):
        self._benchmark_tf_zeros((2, 2), dtypes.bool)

    def benchmark_tf_zeros_2_by_2_string_CPU(self):
        self._benchmark_tf_zeros((2, 2), dtypes.string)

    def benchmark_tf_zeros_2_by_2_float32_GPU(self):
        self._benchmark_tf_zeros((2, 2), dtypes.float32, device=GPU)

    def benchmark_tf_zeros_2_by_2_bool_GPU(self):
        self._benchmark_tf_zeros((2, 2), dtypes.bool, device=GPU)

    def benchmark_tf_zeros_30_by_30_float32_CPU(self):
        self._benchmark_tf_zeros((30, 30), dtypes.float32)

    def benchmark_tf_zeros_30_by_30_bool_CPU(self):
        self._benchmark_tf_zeros((30, 30), dtypes.bool)

    def benchmark_tf_zeros_30_by_30_string_CPU(self):
        self._benchmark_tf_zeros((30, 30), dtypes.string)

    def benchmark_tf_zeros_30_by_30_float32_GPU(self):
        self._benchmark_tf_zeros((30, 30), dtypes.float32, device=GPU)

    def benchmark_tf_zeros_30_by_30_bool_GPU(self):
        self._benchmark_tf_zeros((30, 30), dtypes.bool, device=GPU)

    def benchmark_tf_zeros_100_by_100_float32_CPU(self):
        self._benchmark_tf_zeros((100, 100), dtypes.float32)

    def benchmark_tf_zeros_100_by_100_bool_CPU(self):
        self._benchmark_tf_zeros((100, 100), dtypes.bool)

    def benchmark_tf_zeros_100_by_100_string_CPU(self):
        self._benchmark_tf_zeros((100, 100), dtypes.string)

    def benchmark_tf_zeros_100_by_100_float32_GPU(self):
        self._benchmark_tf_zeros((100, 100), dtypes.float32, device=GPU)

    def benchmark_tf_zeros_100_by_100_bool_GPU(self):
        self._benchmark_tf_zeros((100, 100), dtypes.bool, device=GPU)

    def _benchmark_tf_zeros_like(self, m, device=CPU):
        with context.device(device):

            def func():
                return array_ops.zeros_like(m)

            self._run(func, 3000)

    def benchmark_tf_zeros_like_CPU(self):
        self._benchmark_tf_zeros_like(self._m_2_by_2)

    def benchmark_tf_zeros_like_GPU(self):
        self._benchmark_tf_zeros_like(self._m_2_by_2, device=GPU)

    def benchmark_tf_zeros_like_variable_CPU(self):
        m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
        self._benchmark_tf_zeros_like(m)

    def benchmark_tf_zeros_like_variable_GPU(self):
        m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
        self._benchmark_tf_zeros_like(m, device=GPU)

    def _benchmark_tf_random_uniform_2_by_2(
        self, shape=(2, 2), dtype=dtypes.int32, device=CPU
    ):
        with context.device(device):

            def func():
                return random_ops.random_uniform(shape, maxval=3, dtype=dtype)

            self._run(func, num_iters=self._num_iters_2_by_2)

    def benchmark_tf_random_uniform_2_by_2_integer_CPU(self):
        self._benchmark_tf_random_uniform_2_by_2()

    def benchmark_tf_random_uniform_2_by_2_integer_GPU(self):
        self._benchmark_tf_random_uniform_2_by_2(device=GPU)

    def benchmark_tf_random_uniform_2_by_2_float_CPU(self):
        self._benchmark_tf_random_uniform_2_by_2(dtype=dtypes.float32)

    def benchmark_tf_random_uniform_2_by_2_float_GPU(self):
        self._benchmark_tf_random_uniform_2_by_2(dtype=dtypes.float32, device=GPU)

    def benchmark_tf_random_uniform_2_by_2_default_setting_CPU(self):
        with context.device(CPU):

            def func():
                return random_ops.random_uniform((2, 2))

            self._run(func, num_iters=self._num_iters_2_by_2)

    def benchmark_tf_random_uniform_2_by_2_default_setting_GPU(self):
        with context.device(GPU):

            def func():
                return random_ops.random_uniform((2, 2))

            self._run(func, num_iters=self._num_iters_2_by_2)

    def _benchmark_tf_dropout_2_by_2(
        self, is_rate_tensor=True, noise_shape=None, device=CPU
    ):
        if is_rate_tensor:
            rate = constant_op.constant(0.5, dtype=dtypes.float32)
        else:
            rate = 0.5
        with context.device(device):

            def func():
                return nn_ops.dropout(
                    self._m_2_by_2, rate=rate, noise_shape=noise_shape
                )

            self._run(func, num_iters=self._num_iters_2_by_2)

    def benchmark_tf_dropout_scalar_rate_2_by_2_CPU(self):
        self._benchmark_tf_dropout_2_by_2(is_rate_tensor=False)

    def benchmark_tf_dropout_scalar_rate_2_by_2_GPU(self):
        self._benchmark_tf_dropout_2_by_2(is_rate_tensor=False, device=GPU)

    def benchmark_tf_dropout_2_by_2_CPU(self):
        self._benchmark_tf_dropout_2_by_2()

    def benchmark_tf_dropout_2_by_2_GPU(self):
        self._benchmark_tf_dropout_2_by_2(device=GPU)

    def _benchmark_transpose(
        self, m, num_iters, perm=None, conjugate=False, execution_mode=None
    ):
        def func():
            return array_ops.transpose(m, perm, conjugate)

        self._run(func, num_iters, execution_mode=execution_mode)

    def benchmark_tf_transpose_2_by_2_CPU(self):
        with context.device(CPU):
            m = self._m_2_by_2.cpu()
            self._benchmark_transpose(m, num_iters=self._num_iters_2_by_2)

    def benchmark_tf_transpose_2_by_2_GPU(self):
        with context.device(GPU):
            m = self._m_2_by_2.gpu()
            self._benchmark_transpose(m, num_iters=self._num_iters_2_by_2)

    def benchmark_tf_transpose_variable_2_by_2_CPU(self):
        with context.device(CPU):
            m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
            self._benchmark_transpose(m, num_iters=self._num_iters_2_by_2)

    def benchmark_tf_transpose_variable_2_by_2_GPU(self):
        with context.device(GPU):
            m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
            self._benchmark_transpose(m, num_iters=self._num_iters_2_by_2)

    def benchmark_defun_without_signature(self):
        def func(t1, t2, t3, t4, t5, t6, t7, t8):
            del t1, t2, t3, t4, t5, t6, t7, t8
            return None

        defined = function.defun(func)
        t = constant_op.constant(0.0)

        def cache_computation():
            return defined(t, t, t, t, t, t, t, t)

        self._run(cache_computation, 30000)

    def benchmark_defun_without_signature_and_with_kwargs(self):
        def func(t1, t2, t3, t4, t5, t6, t7, t8):
            del t1, t2, t3, t4, t5, t6, t7, t8
            return None

        defined = function.defun(func)
        t = constant_op.constant(0.0)

        def cache_computation():
            return defined(t1=t, t2=t, t3=t, t4=t, t5=t, t6=t, t7=t, t8=t)

        self._run(cache_computation, 30000)

    def benchmark_defun_with_signature(self):
        def func(t1, t2, t3, t4, t5, t6, t7, t8):
            del t1, t2, t3, t4, t5, t6, t7, t8
            return None

        defined = function.defun(
            func, input_signature=[tensor_spec.TensorSpec([], dtypes.float32)] * 8
        )
        t = constant_op.constant(0.0)

        def signature_computation():
            return defined(t, t, t, t, t, t, t, t)

        self._run(signature_computation, 30000)

    def benchmark_defun_with_signature_and_kwargs(self):
        def func(t1, t2, t3, t4, t5, t6, t7, t8):
            del t1, t2, t3, t4, t5, t6, t7, t8
            return None

        defined = function.defun(
            func, input_signature=[tensor_spec.TensorSpec([], dtypes.float32)] * 8
        )
        t = constant_op.constant(0.0)

        def signature_computation():
            return defined(t1=t, t2=t, t3=t, t4=t, t5=t, t6=t, t7=t, t8=t)

        self._run(signature_computation, 30000)

    def benchmark_matmul_read_variable_op_2_by_2_CPU(self):
        with context.device(CPU):
            m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
            self._benchmark_matmul_read_variable(m, num_iters=self._num_iters_2_by_2)

    def benchmark_matmul_read_variable_op_with_tape_2_by_2_CPU(self):
        with context.device(CPU):
            m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
            self._benchmark_matmul_read_variable_with_tape(
                m, num_iters=self._num_iters_2_by_2
            )

    def benchmark_read_variable_op_2_by_2_CPU(self):
        with context.device(CPU):
            m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
            self._benchmark_read_variable(m, num_iters=self._num_iters_2_by_2)

    def benchmark_read_variable_op_2_by_2_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = resource_variable_ops.ResourceVariable(self._m_2_by_2.gpu())
            self._benchmark_read_variable(m, num_iters=self._num_iters_2_by_2)

    def benchmark_read_variable_op_with_tape_2_by_2_CPU(self):
        with context.device(CPU):
            m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
            self._benchmark_read_variable_with_tape(m, num_iters=self._num_iters_2_by_2)

    def benchmark_read_variable_op_with_tape_2_by_2_GPU(self):
        if not context.num_gpus():
            return
        with context.device(GPU):
            m = resource_variable_ops.ResourceVariable(self._m_2_by_2.gpu())
            self._benchmark_read_variable_with_tape(m, num_iters=self._num_iters_2_by_2)

    def benchmarkScan(self):
        elems = math_ops.range(1600)

        def scan():
            return functional_ops.scan(lambda a, x: a + x, elems, parallel_iterations=1)

        self._run(scan, 100)

    def benchmarkScanDefun(self):
        elems = math_ops.range(1600)

        @function.defun
        def scan():
            return functional_ops.scan(lambda a, x: a + x, elems, parallel_iterations=1)

        self._run(scan, 100)

    def benchmark_fastpath_conversion_type_inference(self):
        c = constant_op.constant(1.0, dtype=dtypes.float32)

        def fn():
            return gen_math_ops.add(c, 1)

        self._run(fn, 10000)

    def benchmark_convert_3x_list_to_tensor(self):
        xs = [1, 2, 3]
        self._run(lambda: ops.convert_to_tensor(xs), 1000)

    def benchmark_convert_3x_array_to_tensor(self):
        xs = np.array([1, 2, 3], dtype=np.int32)
        self._run(lambda: ops.convert_to_tensor(xs), 1000)

    def benchmark_constant_40x2_list_to_tensor(self):
        xs = [[0] * 2] * 40
        self._run(lambda: constant_op.constant(xs), 1000)

    def benchmark_constant_40x2_array_to_tensor(self):
        xs = np.array([[0] * 2] * 40, dtype=np.int32)
        self._run(lambda: constant_op.constant(xs), 1000)

    def benchmark_constant_40x_list_of_2x_arrays_to_tensor(self):
        xs = [np.array([0] * 2, dtype=np.int32)] * 40
        self._run(lambda: constant_op.constant(xs), 1000)

    def benchmark_constant_20x20x20_double_list_to_float32_tensor(self):
        xs = [[[np.linspace(0, 1, 21).tolist()] * 20] * 20]
        self._run(lambda: constant_op.constant(xs, dtype=dtypes.float32), 10000)

    def benchmark_constant_20x20x20_double_list_to_float64_tensor(self):
        xs = [[[np.linspace(0, 1, 21).tolist()] * 20] * 20]
        self._run(lambda: constant_op.constant(xs, dtype=dtypes.float64), 10000)

    def _benchmarkFunctionWithResourceInputs(self, num_resources, num_iters):
        @def_function.function
        def add_all(*args):
            return math_ops.add_n(*args)

        with context.device(CPU):
            resources = []
            for _ in range(num_resources):
                resources.append(resource_variable_ops.ResourceVariable(self._m_2))
            self._run(lambda: add_all(resources), num_iters)

    def benchmarkFunctionWithFiveResourceInputs(self):
        self._benchmarkFunctionWithResourceInputs(5, 1000)

    def benchmarkFunctionWithFiveHundredResourceInputs(self):
        self._benchmarkFunctionWithResourceInputs(500, 100)

    def _benchmarkResourceReadsInCondInInnerFunc(self, var_count):
        rvars = []
        for _ in range(var_count):
            rvars.append(resource_variable_ops.ResourceVariable(1.0))

        # Note: We want to benchmark the graph building time so we intentionally
        # add this outer function so that the tf.function gets retraced every time.
        def benchmark_fn():
            @def_function.function
            def fn_with_many_reads():
                @def_function.function
                def fn_with_many_reads_inner():
                    def then_branch():
                        return math_ops.add_n(rvars)

                    def else_branch():
                        return 0.0

                    return control_flow_ops.cond(
                        constant_op.constant(True), then_branch, else_branch
                    )

                return fn_with_many_reads_inner()

            return fn_with_many_reads()

        with context.device(CPU):
            self._run(benchmark_fn, 10)

    def benchmarkTenThousandResourceReadsInCondInInnerFunc(self):
        self._benchmarkResourceReadsInCondInInnerFunc(10000)

    def benchmarkHundredResourceReadsInCondInInnerFunc(self):
        self._benchmarkResourceReadsInCondInInnerFunc(100)

    def benchmarkTenResourceReadsInCondInInnerFunc(self):
        self._benchmarkResourceReadsInCondInInnerFunc(10)


if __name__ == "__main__":
    test.main()
