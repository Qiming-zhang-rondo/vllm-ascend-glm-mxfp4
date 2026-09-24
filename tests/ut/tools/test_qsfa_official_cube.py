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


COPY_CPP = r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define QSFA_Q8C4_CANDIDATE 1
#include "q8c4_layout.h"
using Q_T=uint16_t;
enum class HardEvent { MTE3_V };
static uint32_t copies=0, sets=0, waits=0;
template<HardEvent E> void SetFlag(int){assert(sets==waits);++sets;}
template<HardEvent E> void WaitFlag(int){assert(sets==waits+1);++waits;}
struct Pipe {
    template<HardEvent E> int FetchEventID(){return 0;}
};
template<class T> struct LocalTensor {
    uint8_t* ptr;
    LocalTensor operator[](uint32_t n) const{return {ptr+n*sizeof(T)};}
};
struct Buffer {
    uint8_t* ptr;
    template<class T> LocalTensor<T> GetTensor(){return {ptr};}
    template<class T> LocalTensor<T> Get(){return {ptr};}
};
struct DataCopyParams {uint32_t blockCount,blockLen,srcGap,dstGap;};
template<class T> void DataCopy(LocalTensor<T> dst,LocalTensor<T> src,DataCopyParams p){
    ++copies;
    for(uint32_t b=0;b<p.blockCount;++b)
        std::memcpy(dst.ptr+b*(p.blockLen+p.dstGap)*32,
                    src.ptr+b*(p.blockLen+p.srcGap)*32,p.blockLen*32);
}
uint32_t Align16Func(uint32_t x){return (x+15)/16*16;}
void actual_copy(uint8_t* destination,uint8_t* decoded,uint8_t* mxSource,uint32_t start){
    Buffer outputL1{destination},mxCacheStage{mxSource};
    LocalTensor<Q_T> antiKvTensorAsB16{decoded};
    const int64_t dealRow=16,s2StartIdx=start;
    struct {uint32_t s2RealSize;} runInfo{64};
    struct {uint32_t dSizeNope,dSizeRope;} constInfo{512,64};
    Pipe pipe;auto* tPipe=&pipe;
    __ACTUAL_COPY_BODY__
}
int main(){
    using namespace qsfa_q8c4_layout;
    constexpr uint32_t guard=64;
    std::vector<uint8_t> actual(KV_BYTES+2*guard,0x5A),expected=actual;
    std::vector<uint8_t> decoded(576*17*2),mx(8448);
    for(uint32_t row:{0U,16U,32U,48U}){
        for(uint32_t i=0;i<decoded.size();++i)decoded[i]=(i*37+row*5)%256;
        for(uint32_t i=0;i<mx.size();++i)mx[i]=(i*17+row*11)%256;
        copies=0;
        actual_copy(actual.data()+guard,decoded.data(),mx.data(),row);
        assert(copies==3);
        // Original nine-copy behavior, evaluated independently in bytes.
        for(uint32_t col=0;col<36;++col)
            std::memcpy(expected.data()+guard+(col*64+row)*32,
                        decoded.data()+col*17*32,16*32);
        for(uint32_t chunk=0;chunk<4;++chunk){
            for(uint32_t col=0;col<4;++col)
                std::memcpy(expected.data()+guard+K_DATA+chunk*64*128+(col*64+row)*32,
                            mx.data()+chunk*16*128+col*16*32,16*32);
            std::memcpy(expected.data()+guard+K_SCALE+chunk*64*4+row*4,
                        mx.data()+8192+chunk*64,64);
        }
        assert(actual==expected); // Includes untouched rows and both guards.
    }
    assert(sets==4 && waits==4); // Same single-buffer lifetime fence per call.
}
"""


def test_actual_coalesced_copy_matches_original_bytes_and_fences(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    header = OFFICIAL / (
        "vendor/attention/kv_quant_sparse_flash_attention/op_kernel/arch35/"
        "kv_quant_sparse_flash_attention_service_vector_mla_arch35.h"
    )
    # Compile the implementation body itself; a changed copy descriptor is
    # tested rather than merely comparing two handwritten layout formulas.
    implementation = header.read_text().split("::CopyOutKvUb2L1(", 1)[1].split("\nTEMPLATES_DEF_NO_DEFAULT", 1)[0]
    body = implementation[implementation.index("{") + 1 : implementation.rindex("}")]
    source, binary = tmp_path / "copy.cpp", tmp_path / "copy"
    source.write_text(COPY_CPP.replace("__ACTUAL_COPY_BODY__", body))
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(OFFICIAL), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


PV_CPP = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <vector>
#define __aicore__
struct bfloat16_t { uint16_t bits; };
struct fp8_e8m0_t {};
struct fp8_e4m3fn_t {};
struct fp8_e5m2_t {};
struct hifloat8_t {};
template<class A,class B> using IsSameType=std::is_same<A,B>;
constexpr uint32_t K_STEP_ALIGN_BASE=2,M_STEP_ALIGN_BASE=2,FP16_ONE_FRACTAL_ELEMENT=16;
uint32_t CeilAlign(uint32_t n,uint32_t a){return (n+a-1)/a*a;}
enum class ABLayout { MK,KN };
enum class HardEvent { M_MTE1,MTE1_M };
struct LoadData2DParamsV2 {
    uint32_t mStartPosition=0,kStartPosition=0,mStep=0,kStep=0,srcStride=0,dstStride=0;
    bool ifTranspose=false;
};
struct MmadParams {
    uint32_t m=0,n=0,k=0,unitFlag=0;
    bool cmatrixInitVal=false,cmatrixSource=false;
};
enum Space { L1,L0A,L0B,L0C };
template<class T> struct LocalTensor {
    Space space=L1; uint64_t offset=0,end=0;
    LocalTensor operator[](uint64_t n)const{
        assert(offset+n*sizeof(T)<=end);
        return {space,offset+n*sizeof(T),end};
    }
};
struct LoadRecord { Space space; uint64_t src; LoadData2DParamsV2 p; };
static std::vector<LoadRecord> loads;
static std::vector<uint32_t> written;
static uint32_t activeM,activeK,mmads;
template<class T> void LoadData(LocalTensor<T> dst,LocalTensor<T> src,LoadData2DParamsV2 p){
    static_assert(std::is_same_v<T,bfloat16_t>);
    assert(src.space==L1 && (dst.space==L0A || dst.space==L0B));
    assert(p.mStartPosition==0 && p.kStartPosition==0);
    uint64_t sourceBytes=((p.kStep-1)*p.srcStride+p.mStep)*512;
    uint64_t destinationBytes=(p.ifTranspose ? (p.mStep-1)*p.dstStride+p.kStep
                                                           : (p.kStep-1)*p.dstStride+p.mStep)*512;
    assert(src.offset+sourceBytes<=src.end);
    assert(dst.offset+destinationBytes<=dst.end);
    if(dst.space==L0A){
        assert(!p.ifTranspose && p.mStep==CeilAlign(activeM,16)/16);
        assert(p.kStep==activeK/16 && p.srcStride==p.mStep && p.dstStride==p.mStep);
    } else {
        assert(p.ifTranspose && p.mStep==activeK/16 && p.kStep==8);
        assert(p.srcStride==activeK/16 && p.dstStride==8);
    }
    loads.push_back({dst.space,src.offset,p});
}
template<class T> void Mmad(LocalTensor<float> c,LocalTensor<T> a,LocalTensor<T> b,MmadParams p){
    static_assert(std::is_same_v<T,bfloat16_t>);
    assert(c.space==L0C && a.space==L0A && b.space==L0B);
    assert(p.m==activeM && p.n==128 && p.k==activeK);
    assert(p.cmatrixInitVal && !p.cmatrixSource && p.unitFlag==0);
    uint64_t elems=CeilAlign(p.m,16)*p.n;
    assert(c.offset+elems*sizeof(float)<=c.end);
    for(uint64_t i=c.offset/sizeof(float);i<c.offset/sizeof(float)+elems;++i){
        assert(i<written.size()); ++written[i];
    }
    ++mmads;
}
struct State { int free=1,ready=0; };
struct MockBuffer {
    State* state; Space space; uint64_t offset,bytes;
    template<HardEvent E> void Wait(){
        int& value=E==HardEvent::M_MTE1?state->free:state->ready;
        assert(value==1); --value;
    }
    template<HardEvent E> void Set(){
        int& value=E==HardEvent::M_MTE1?state->free:state->ready;
        assert(value==0); ++value;
    }
    template<class T> LocalTensor<T> GetTensor(uint64_t index=0){
        assert(index*sizeof(T)<=bytes);
        return {space,offset+index*sizeof(T),offset+bytes};
    }
};
struct Policy {
    State states[2]; uint32_t next=0; Space space; uint32_t bytes;
    MockBuffer Get(){uint32_t i=next; next^=1;return {&states[i],space,i*bytes,bytes};}
};
__ACTUAL_MM_PARAM__
__ACTUAL_GET_LOAD_K__
__ACTUAL_GET_BLOCK_NUM__
__ACTUAL_LOAD_A__
__ACTUAL_LOAD_B__
__ACTUAL_MATMUL_N__

template<uint32_t S2> void run(uint32_t heads){
    using Q_T=bfloat16_t; using T=float;
    constexpr uint32_t s1BaseSize=64,s2BaseSize=S2,dBaseMatmulSize=128;
    struct {uint32_t mRealSize,s2RealSize;} runInfo{heads,S2};
    struct {uint32_t dSizeNope;} constInfo{512};
    // Both paths put P directly after BF16 V, at the old RoPE offset.
    MockBuffer inputRightBuf{nullptr,L1,0,S2*576*2};
    MockBuffer mm2ResL0C{nullptr,L0C,0,131072};
    Policy mmL0ABuffers{{},0,L0A,16384},mmL0BBuffers{{},0,L0B,32768};
    activeM=heads; activeK=S2; mmads=0; loads.clear();
    written.assign(131072/sizeof(float),0);
    __ACTUAL_PV_CALL__
    assert(mmads==4 && loads.size()==5);
    assert(loads[0].space==L0A && loads[0].src==S2*512*2);
    for(uint32_t n=0;n<4;++n){
        assert(loads[n+1].space==L0B);
        assert(loads[n+1].src==n*S2*128*2);
    }
    // Four adjacent N128 slices cover the full padded M x 512 C matrix once;
    // padding outside it remains untouched. All actual helper loads fit L0.
    const uint32_t count=CeilAlign(heads,16)*512;
    for(uint32_t i=0;i<written.size();++i)assert(written[i]==(i<count?1U:0U));
    for(auto state:mmL0ABuffers.states)assert(state.free==1 && state.ready==0);
    for(auto state:mmL0BBuffers.states)assert(state.free==1 && state.ready==0);
}
int main(){
    for(uint32_t heads:{8U,16U,32U})run<64>(heads);
    for(uint32_t heads:{8U,16U,32U,64U})run<128>(heads);
}
"""


def _cpp_definition(text, marker, template=""):
    """Extract a source definition through its matching closing brace."""
    start = text.index(marker)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    return template + text[start:end]


def test_actual_pv_n128_calls_cover_output_and_fit_existing_l0_buffers(tmp_path):
    """Execute actual PV caller/MatmulN/load descriptors with bounded host mocks."""
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    matmul = (OFFICIAL / "vendor/attention/common/op_kernel/matmul.h").read_text()
    cube = (
        OFFICIAL / "vendor/attention/kv_quant_sparse_flash_attention/op_kernel/arch35/"
        "kv_quant_sparse_flash_attention_service_cube_mla_arch35.h"
    ).read_text()
    pv_caller = cube.split("::IterateBmm2QSFA(", 1)[1]
    pv_call = pv_caller[pv_caller.index("MMParam qsfaParam") : pv_caller.index("inputRightBuf.SetCrossCore()")]
    matmul_n = matmul[matmul.index("// 切N\n") : matmul.index("// 切M\n")]
    replacements = {
        "__ACTUAL_MM_PARAM__": _cpp_definition(matmul, "struct MMParam") + ";",
        "__ACTUAL_GET_LOAD_K__": _cpp_definition(matmul, "__aicore__ inline uint32_t GetLoadK("),
        "__ACTUAL_GET_BLOCK_NUM__": _cpp_definition(
            matmul, "__aicore__ inline uint32_t GetBlockNum(", "template<class T>\n"
        ),
        "__ACTUAL_LOAD_A__": _cpp_definition(matmul, "__aicore__ inline void LoadDataToL0A(", "template<class T>\n"),
        "__ACTUAL_LOAD_B__": _cpp_definition(matmul, "__aicore__ inline void LoadDataToL0B(", "template<class T>\n"),
        "__ACTUAL_MATMUL_N__": matmul_n,
        "__ACTUAL_PV_CALL__": pv_call,
    }
    code = PV_CPP
    for marker, replacement in replacements.items():
        code = code.replace(marker, replacement)
    source, binary = tmp_path / "pv.cpp", tmp_path / "pv"
    source.write_text(code)
    # Upstream's generic templates have unused parameters on the BF16 branch.
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-parameter",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
