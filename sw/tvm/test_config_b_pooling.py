# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Testing topi pooling operator for VTA"""

import json
import os

import pytest
import numpy as np
from collections import namedtuple

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.contrib import util
from tvm.contrib.pickle_memoize import memoize
import topi
import topi.testing
from topi.util import get_const_tuple
import vta
from vta import program_fpga, reconfig_runtime
import vta.testing
from vta.testing import simulator


Workload = namedtuple("Conv2DWorkload",
                      ['batch', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

# Get batch info from env
env = vta.get_env()

# CoDeNet workloads
pooling_wkls = [
    ('config_b_pooling',  Workload(env.BATCH, 128, 128, 24,   24,  3, 3, 1, 1, 2, 2)),
]

# FIXME: we need a custom clip operator to circumvent a pattern detection limitation
@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x

def run_pooling(env, remote, wl, target,
               check_correctness=True, print_ir=False,
               samples=10):

    # Workload assertions
    assert wl.hpad == wl.wpad
    pool_type = 'max'

    # Perform packing only if we are targeting the accelerator
    if "arm_cpu" in target.keys:
        data_pack = False
        layout = "NCHW"
        #pooling_fcompute = topi.arm_cpu.pooling_nchw_spatial_pack
        pooling_fcompute =  topi.nn.pool
        #pooling_fschedule = topi.arm_cpu.schedule_pooling_nchw_spatial_pack
        pooling_fschedule = topi.generic.schedule_pool
    elif "vta" in target.keys:
        data_pack = True
        layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN)
        pooling_fcompute = vta.top.pooling_packed
        pooling_fschedule = vta.top.schedule_pooling_packed

    # Derive shapes depending upon packing
    a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
    w_shape = (wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)
    # output shape
    b_shape = (wl.batch, wl.out_filter, 1, 1)
    if data_pack:
        data_shape = (wl.batch//env.BATCH, wl.in_filter//env.BLOCK_IN,
                      wl.height, wl.width, env.BATCH, env.BLOCK_IN)
        kernel_shape = (wl.out_filter//env.BLOCK_OUT, wl.in_filter//env.BLOCK_IN,
                        wl.hkernel, wl.wkernel, env.BLOCK_OUT, env.BLOCK_IN)
        bias_shape = (wl.batch//env.BATCH, wl.out_filter//env.BLOCK_OUT,
                      1, 1, env.BATCH, env.BLOCK_OUT)
    else:
        data_shape = a_shape
        kernel_shape = w_shape
        bias_shape = b_shape
    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    bias = te.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)
    padding = relay.nn.get_pad_tuple2d((wl.hpad, wl.wpad))

    # Define base computation schedule
    with target:
        res = topi.nn.pool(data, kernel=[3, 3], stride=[2, 2], padding=padding,
                      pool_type=pool_type,
                      layout="NCHW")
 #       res = topi.right_shift(res, 8)
 #       res = topi.add(res, bias)
 #       res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
 #       res = topi.cast(res, env.out_dtype)
        # Derive base schedule
        s = pooling_fschedule([res], layout)
        if print_ir:
            print(vta.lower(s, [data, kernel, bias, res], simple_mode=True))
    # get output shape
    _, oc, oh, ow = get_const_tuple(res.shape)
    # Derive number of ops
    fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
    fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
    num_ops = 2 * wl.batch * fout_height * fout_width * wl.hkernel * wl.wkernel * wl.out_filter * wl.in_filter

    # @memoize("vta.tests.test_benchmark_topi.pooling.verify_nchw")
    def get_ref_data():
        # derive min max for act, wgt, and bias types (max non inclusive)
        a_min, a_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        b_min, b_max = 0 - 1 << (env.INP_WIDTH + env.WGT_WIDTH - 2), 1 << (env.INP_WIDTH + env.WGT_WIDTH - 2)
        a_np = np.random.randint(a_min, a_max, size=a_shape).astype(data.dtype)

        pad_shape = (wl.batch, wl.in_filter, wl.height+wl.hpad*2, wl.width+wl.wpad*2)
        pad_np = np.zeros(shape=pad_shape).astype(data.dtype)
        no_zero = (range(wl.batch), range(wl.in_filter), (range(wl.hpad, wl.height+wl.hpad)), (range(wl.wpad, wl.width+wl.wpad)))
        pad_np[np.ix_(*no_zero)] = a_np
        b_shape = (wl.batch, oc, oh, ow)
        b_np = np.random.randint(b_min, b_max, size=b_shape).astype(env.acc_dtype)
        kw, kh = 3, 3
        sw, sh = 2, 2
        for i in range(oh):
            for j in range(ow):
                b_np[:, :, i, j] = np.max(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2, 3))
        b_np = np.maximum(b_np, 0.0)
        return a_np, pad_np, b_np

    # Data in original format
    data_np, _,res_ref = get_ref_data()

    # Build
    if "vta" in target.keys:
        mod = vta.build(s, [data,  res],
                        target=target,
                        target_host=env.target_host,
                        name="pooling")
    else:
        mod = tvm.build(s, [data,  res],
                        target=target,
                        target_host=env.target_host,
                        name="pooling")
    temp = util.tempdir()
    mod.save(temp.relpath("pooling.o"))
    remote.upload(temp.relpath("pooling.o"))
    f = remote.load_module("pooling.o")
    ctx = remote.context(str(target))

    res_np = np.zeros(topi.util.get_const_tuple(res.shape)).astype(res.dtype)
    data_arr = tvm.nd.array(data_np, ctx)
    res_arr = tvm.nd.array(res_np, ctx)
    time_f = f.time_evaluator("pooling", ctx, number=samples)

    # In vta sim mode, collect simulator runtime statistics
    stats = {}
    cost = None
    if env.TARGET in ["sim", "tsim"]:
        # Check if we're in local RPC mode (allows us to rebuild the
        # runtime on the fly when varying the VTA designs)
        local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
        if local_rpc:
            if env.TARGET == "sim":
                remote.get_function("vta.simulator.profiler_clear")()
            else:
                remote.get_function("vta.tsim.profiler_clear")()
            cost = time_f(data_arr, res_arr)
            if env.TARGET == "sim":
                stats = json.loads(remote.get_function("vta.simulator.profiler_status")())
            else:
                stats = json.loads(remote.get_function("vta.tsim.profiler_status")())
        else:
            simulator.clear_stats()
            cost = time_f(data_arr, res_arr)
            stats = simulator.stats()
    else:
        cost = time_f(data_arr, res_arr)
        print(cost)

    # Check correctness
    correct = False
    if check_correctness:
        res_orig = res_arr.asnumpy()
        res_orig = np.maximum(res_orig, 0.0)
        res_ref = res_ref.astype(env.out_dtype)
        res_orig = res_orig.astype(env.out_dtype)
        correct = np.allclose(res_orig, res_ref)

    gops = (num_ops / cost.mean) / float(10 ** 9)
    status = "PASSED" if correct else "FAILED"
    if "arm_cpu" in target.keys:
        device = "CPU"
    elif "vta" in target.keys:
        device = "VTA"
    print("%s POOLING TEST %s: Time cost = %g sec/op" % (device, status, cost.mean))

    return correct, cost, stats

@pytest.mark.parametrize("device", ["vta", "arm_cpu"])
def test_pooling(device):
    def _run(env, remote):
        if device == "vta":
            target = env.target
            if env.TARGET not in ["sim", "tsim"]:
                assert tvm.runtime.enabled("rpc")
                program_fpga(remote, bitstream=None)
                reconfig_runtime(remote)
        elif device == "arm_cpu":
            target = env.target_vta_cpu
        with autotvm.tophub.context(target): # load pre-tuned schedule parameters
            for _, wl in pooling_wkls:
                print(wl)
                run_pooling(env, remote, wl, target)
    vta.testing.run(_run)

if __name__ == "__main__":
    test_pooling(device="arm_cpu")
