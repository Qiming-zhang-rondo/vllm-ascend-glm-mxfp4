"""Host checks for the actual candidate MX-QK helper, not an A5 execution claim."""

import shutil
import subprocess
from pathlib import Path

import pytest

OFFICIAL = Path(__file__).resolve().parents[3] / "benchmarks/qsfa_q8c4_o8/official"

CPP = r"""
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <vector>
#define __aicore__
struct fp8_e4m3fn_t { uint8_t value; };
struct fp8_e8m0_t { uint8_t value; };
struct mx_fp8_e4m3_t { uint8_t value; };
struct bfloat16_t { uint16_t value; };
enum class HardEvent { M_MTE1, MTE1_M };
struct LoadData2DParamsV2 {
    uint32_t mStep=0, kStep=0, srcStride=0, dstStride=0;
    bool ifTranspose=false;
};
struct LoadData2DMxParams {
    uint32_t xStep=0, yStep=0, srcStride=0, dstStride=0;
};
struct MmadParams { uint32_t m=0,n=0,k=0; bool cmatrixInitVal=false,cmatrixSource=false; };
template<class T> struct LocalTensor {
    uint32_t offset, end;
    LocalTensor operator[](uint32_t n) const {
        assert(offset+n*sizeof(T)<=end);
        return {offset+n*static_cast<uint32_t>(sizeof(T)),end};
    }
    template<class U> LocalTensor<U> ReinterpretCast() const {return {offset,end};}
};
struct Load { uint32_t src, scale, m, k, stride; bool mx; };
std::vector<Load> loads;
struct Compute { uint32_t dst,m,n,k; bool mx,init; };
std::vector<Compute> computes;
template<class A, class B> void LoadData(const LocalTensor<A>& dst,
    const LocalTensor<B>& src, const LocalTensor<fp8_e8m0_t>& scale,
    LoadData2DParamsV2 p, LoadData2DMxParams s) {
    static_assert(std::is_same_v<A,mx_fp8_e4m3_t>);
    static_assert(std::is_same_v<B,fp8_e4m3fn_t>);
    assert(!p.ifTranspose && s.xStep==p.mStep && s.yStep==2);
    assert(s.srcStride==2 && s.dstStride==2);
    assert(src.offset + (p.kStep-1)*p.srcStride*512 + p.mStep*512 <= src.end);
    assert(scale.offset + s.xStep*16*4 <= scale.end);
    assert(dst.offset+p.mStep*p.kStep*512 <= dst.end);
    loads.push_back({src.offset,scale.offset,p.mStep,p.kStep,p.srcStride,true});
}
template<class T> void LoadData(const LocalTensor<T>& dst,
    const LocalTensor<T>& src, LoadData2DParamsV2 p) {
    static_assert(std::is_same_v<T,bfloat16_t>);
    assert(!p.ifTranspose);
    assert(src.offset + (p.kStep-1)*p.srcStride*512 + p.mStep*512 <= src.end);
    assert(dst.offset+p.mStep*p.kStep*512 <= dst.end);
    loads.push_back({src.offset,0,p.mStep,p.kStep,p.srcStride,false});
}
template<class T> void Mmad(const LocalTensor<float>& c, const LocalTensor<T>&,
                          const LocalTensor<T>&, MmadParams p) {
    assert(!p.cmatrixSource);
    computes.push_back({c.offset,p.m,p.n,p.k,std::is_same_v<T,mx_fp8_e4m3_t>,p.cmatrixInitVal});
}
struct State { int free=1,ready=0; };
struct Buffer {
    State* state;
    uint32_t offset,bytes;
    template<HardEvent E> void Wait() {
        int& value = E==HardEvent::M_MTE1 ? state->free : state->ready;
        assert(value==1); --value;
    }
    template<HardEvent E> void Set() {
        int& value = E==HardEvent::M_MTE1 ? state->free : state->ready;
        assert(value==0); ++value;
    }
    template<class T> LocalTensor<T> GetTensor(){return {offset,offset+bytes};}
};
struct Policy {
    State state[2]; uint32_t next=0,bytes;
    Buffer Get(){uint32_t i=next;next^=1;return {&state[i],i*bytes,bytes};}
};
#include "q8c4_cube.h"
int main() {
    using namespace qsfa_q8c4_layout;
    assert(3*KV_BYTES+Q_BYTES==364544);
    for(uint32_t h : {8U,16U,32U,64U}) {
        loads.clear(); computes.clear();
        Policy a{{},0,16384},b{{},0,32768};
        BaseApi::QsfaMxQk(LocalTensor<uint8_t>{Q_BASE,L1_BYTES},
            LocalTensor<uint8_t>{KV_BYTES,2*KV_BYTES},a,b,
            LocalTensor<float>{0,131072},h,N);
        assert(loads.size()==10 && computes.size()==5);
        for(uint32_t chunk=0;chunk<4;++chunk) {
            const auto& qa=loads[chunk*2]; const auto& kb=loads[chunk*2+1];
            assert(qa.mx && kb.mx);
            assert(qa.src==Q_BASE+Q_DATA+chunk*M*K);
            assert(qa.scale==Q_BASE+Q_SCALE+chunk*M*4);
            assert(kb.src==KV_BYTES+K_DATA+chunk*N*K);
            assert(kb.scale==KV_BYTES+K_SCALE+chunk*N*4);
            assert(qa.m==(h+15)/16 && kb.m==4);
            assert(qa.stride==M/16 && kb.stride==N/16);
        }
        assert(loads[8].src==Q_BASE+Q_ROPE && !loads[8].mx);
        assert(loads[9].src==KV_BYTES+ROPE && !loads[9].mx);
        for(uint32_t i=0;i<5;++i) {
            auto c=computes[i];
            assert(c.dst==0 && c.m==h && c.n==N);
            assert(c.mx==(i<4) && c.k==(i<4?128U:64U));
            assert(c.init==(i==0));
        }
        for(auto s : a.state) assert(s.free==1 && s.ready==0);
        for(auto s : b.state) assert(s.free==1 && s.ready==0);
    }
}
"""


def test_actual_mx_cube_loads_scales_fp32_accumulation_and_buffer_events(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "cube.cpp"
    binary = tmp_path / "cube"
    source.write_text(CPP)
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(OFFICIAL), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
