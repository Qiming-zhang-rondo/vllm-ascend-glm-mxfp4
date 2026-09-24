// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "acl/acl.h"

extern "C" void qsfa_gather_launch(
    aclrtStream stream, void* q, void* qs, void* kv, void* ks, void* rope, void* indices,
    void* qn, void* qns, void* qr, void* kn, void* kns, void* kr, void* vt, void* status,
    uint32_t h, uint32_t m, uint32_t k, uint32_t s);
extern "C" void qsfa_qk_launch(
    aclrtStream stream, void* qn, void* kn, void* qns, void* kns, void* qr, void* kr,
    void* scores, uint32_t m, uint32_t s);
extern "C" void qsfa_softmax_launch(
    aclrtStream stream, void* scores, void* p, uint32_t m, uint32_t s, float scale);
extern "C" void qsfa_pv_launch(
    aclrtStream stream, void* p, void* vt, void* acc, uint32_t m, uint32_t s);
extern "C" void qsfa_output_launch(
    aclrtStream stream, void* acc, void* out, void* oscale, uint32_t h);
extern "C" void qsfa_qk_tiled_launch(
    aclrtStream stream, void* q, void* qs, void* kv, void* ks, void* rope, void* indices,
    void* scores, void* status, uint32_t h, uint32_t m, uint32_t k, uint32_t s);
extern "C" void qsfa_pv_tiled_launch(
    aclrtStream stream, void* p, void* kv, void* ks, void* indices, void* out, void* oscale,
    uint32_t h, uint32_t m, uint32_t k, uint32_t s);
