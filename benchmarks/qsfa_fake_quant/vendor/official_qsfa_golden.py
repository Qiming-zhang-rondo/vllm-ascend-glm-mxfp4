#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Exact function excerpts from CANN ops-transformer, see SOURCES.json.
# Only imports are reduced to math/torch; function bodies are unchanged.
# ruff: noqa

import math

import torch


def gatherKV(
    k_tensor,
    v_tensor,
    sparse_indices_Indices,
    sparse_blocksize,
    sparse_blockcount,
    batch,
    n2Idx,
    s1Idx,
    curr_actualSeq,
    curr_actualSeq_q,
    sparse_mode,
):
    s2_sparse = list()
    if sparse_mode == 0:
        threshold = curr_actualSeq
    elif sparse_mode == 3:
        delta_s = curr_actualSeq - curr_actualSeq_q
        threshold = delta_s + s1Idx + 1
    validCount = min(sparse_blockcount, math.ceil(threshold / sparse_blocksize))
    for i in range(validCount):
        sparseIndicesId = sparse_indices_Indices[i]

        if sparseIndicesId == -1:
            break
        begin_Idx = sparseIndicesId * sparse_blocksize
        end_Idx = (
            begin_Idx + sparse_blocksize
            if begin_Idx + sparse_blocksize <= curr_actualSeq
            else curr_actualSeq
        )
        if begin_Idx >= threshold:
            continue
        if end_Idx <= threshold:
            s2_sparse.extend(range(begin_Idx, end_Idx))
        else:
            s2_sparse.extend(range(begin_Idx, threshold))

    emptyFlag = False
    if len(s2_sparse) == 0:
        k_sparse, v_sparse = [], []
        emptyFlag = True
    else:
        k_sparse, v_sparse = (
            k_tensor[batch, n2Idx, s2_sparse, :],
            v_tensor[batch, n2Idx, s2_sparse, :],
        )

    return emptyFlag, k_sparse, v_sparse


def softmax(x):
    x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    return ans


def _t_increattention_bnsd(fa_param):
    batch_size = fa_param["b"]
    numheads = fa_param["numHeads"]
    numKeyValueHeads = fa_param["numKeyValueHeads"]
    actualSeqLengths_q = fa_param["actualSeqLengths_q"]
    actualSeqLengths_kv = fa_param["actualSeqLengths_kv"]
    scaleValue = fa_param["scaleValue"]
    sparse_blocksize = fa_param["sparse_blocksize"]
    sparse_blockcount = fa_param["sparse_blockcount"]
    sparseIndicesIndices = fa_param["sparse_indices_bnsd_tensor"]
    out_shape_bnsd = fa_param["q_bnsd_shape"]
    out_shape_bnsd[-1] = fa_param["v_bnsd_shape"][-1]
    sparse_mode = fa_param["sparse_mode"]
    g = numheads // numKeyValueHeads

    q_bnsd_tensor = fa_param["q_bnsd_tensor"]
    k_bnsd_tensor = fa_param["k_bnsd_tensor"].float()
    k_dequant_scale_bnsd_tensor = fa_param["k_dequant_scale_bnsd_tensor"].float()
    v_bnsd_tensor = fa_param["v_bnsd_tensor"].float()
    v_dequant_scale_bnsd_tensor = fa_param["v_dequant_scale_bnsd_tensor"].float()
    tile_size = 128
    if fa_param["q_dtype"] == "float16":
        k_bnsd_tensor[..., :512] = (
            k_bnsd_tensor[..., :512]
            * torch.repeat_interleave(k_dequant_scale_bnsd_tensor, tile_size, dim=-1)
        ).to(torch.float16)
    elif fa_param["q_dtype"] == "bfloat16":
        k_bnsd_tensor[..., :512] = (
            k_bnsd_tensor[..., :512]
            * torch.repeat_interleave(k_dequant_scale_bnsd_tensor, tile_size, dim=-1)
        ).to(torch.bfloat16)
    elif fa_param["q_dtype"] == "float32":
        k_bnsd_tensor[..., :512] = (
            k_bnsd_tensor[..., :512]
            * torch.repeat_interleave(k_dequant_scale_bnsd_tensor, tile_size, dim=-1)
        ).to(torch.float32)
    v_bnsd_tensor = k_bnsd_tensor[..., :512].clone()
    matmul_dtype = torch.float32
    y = torch.zeros(out_shape_bnsd, dtype=torch.float32)

    q_bnsd_tensor = fa_param["q_bnsd_tensor"].float()

    for batch in range(batch_size):
        curr_actualSeq_q = actualSeqLengths_q[batch]
        curr_actualSeq = actualSeqLengths_kv[batch]
        for n2Idx in range(numKeyValueHeads):
            for s1Idx in range(curr_actualSeq_q):
                if s1Idx < curr_actualSeq_q - curr_actualSeq and sparse_mode != 0:
                    y[batch, n2Idx * g : (n2Idx + 1) * g, s1Idx, :] = torch.zeros(
                        [g, out_shape_bnsd[-1]], dtype=torch.float32
                    )
                    continue
                q_curr = q_bnsd_tensor[batch, n2Idx * g : (n2Idx + 1) * g, s1Idx, :]
                sparse_indices_Indices = sparseIndicesIndices[batch, n2Idx, s1Idx, :]

                emptyFlag, k_sparse, v_sparse = gatherKV(
                    k_bnsd_tensor,
                    v_bnsd_tensor,
                    sparse_indices_Indices,
                    sparse_blocksize,
                    sparse_blockcount,
                    batch,
                    n2Idx,
                    s1Idx,
                    curr_actualSeq,
                    curr_actualSeq_q,
                    sparse_mode,
                )
                if emptyFlag:
                    continue
                bmm1Res = torch.matmul(q_curr.float(), k_sparse.float().T)
                scaleRes = bmm1Res * scaleValue
                softmax_res = softmax(scaleRes)
                if fa_param["q_dtype"] == "float16":
                    bmm2Res = torch.matmul(
                        softmax_res.to(torch.float16).float(), v_sparse.float()
                    )
                elif fa_param["q_dtype"] == "bfloat16":
                    bmm2Res = torch.matmul(
                        softmax_res.to(torch.bfloat16).float(), v_sparse.float()
                    )
                else:
                    bmm2Res = torch.matmul(softmax_res.float(), v_sparse.float())
                y[batch, n2Idx * g : (n2Idx + 1) * g, s1Idx, :] = bmm2Res
    return y
