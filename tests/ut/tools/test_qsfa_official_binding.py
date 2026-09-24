"""Compile actual binding with CPU libtorch and execute schemas/Meta kernels.

NPU headers are deliberately throwing host shims. No NPU call, launch, ABI
compatibility with torch_npu, or device allocation is claimed by this test.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import include_paths, library_paths

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "benchmarks/qsfa_q8c4_o8/csrc/official_binding.cpp"
SHIMS = {
    "acl/acl.h": """
#pragma once
using aclrtStream=void*;
inline const char* aclrtGetSocName(){throw std::runtime_error("No ACL runtime in host test");}
""",
    "torch_npu/csrc/core/npu/NPUGuard.h": """
#pragma once
namespace c10_npu {
struct NPUGuard {
    explicit NPUGuard(c10::Device){throw std::runtime_error("No NPU guard in host test");}
};
}
""",
    "torch_npu/csrc/core/npu/NPUStream.h": """
#pragma once
namespace c10_npu {
struct NPUStream { void* stream(bool) const {throw std::runtime_error("No NPU stream in host test");} };
inline NPUStream getCurrentNPUStream(){throw std::runtime_error("No NPU stream in host test");}
}
""",
    "torch_npu/csrc/core/npu/NPUCachingAllocator.h": """
#pragma once
namespace c10_npu::NPUCachingAllocator {
inline void recordStream(const c10::DataPtr&,const NPUStream&) {
    throw std::runtime_error("No NPU allocator in host test");
}
}
""",
}
CPP = r"""
#include <stdexcept>
#include <limits>
#include <iostream>
#include "official_binding.cpp"
extern "C" void qsfa_official_fp8_launch(aclrtStream,void*,void*,void*,void*,void*,void*,void*,void*,
    uint32_t,uint32_t,uint32_t,float){throw std::runtime_error("No device launch in host test");}
extern "C" void qsfa_official_q8c4_launch(aclrtStream,void*,void*,void*,void*,void*,void*,void*,void*,
    uint32_t,uint32_t,uint32_t,float){throw std::runtime_error("No device launch in host test");}

using Result=std::tuple<at::Tensor,at::Tensor,at::Tensor>;
using Signature=Result(const at::Tensor&,const at::Tensor&,const at::Tensor&,
                       const at::Tensor&,const at::Tensor&,const at::Tensor&,double);
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> void rejects(F&& call){
    bool rejected=false;
    try{call();}catch(const c10::Error&){rejected=true;}
    require(rejected,"invalid inputs were accepted");
}
int main(int argc,char** argv){
    try {
        auto low=c10::Dispatcher::singleton().findSchemaOrThrow("qsfa_q8c4_o8::forward_official","").typed<Signature>();
        auto base=c10::Dispatcher::singleton()
            .findSchemaOrThrow("qsfa_q8c4_o8::forward_official_fp8","").typed<Signature>();
        auto byte=at::TensorOptions().device(at::kMeta).dtype(at::kByte);
        auto integer=byte.dtype(at::kInt);
        auto cache=at::empty({32,256,1,416},byte);
        auto table=at::empty({1,32},integer);
        auto idx=at::empty({1,1,2048},integer);
        auto cuq=at::empty({1},integer),len=at::empty({1},integer);
        const std::string mode=argc>1?argv[1]:"valid";
        if(mode=="valid") {
            for(int h:{8,16,32}){
                auto result=low.call(at::empty({h,608},byte),cache,idx,table,cuq,len,0.04);
                const auto& out=std::get<0>(result);const auto& scale=std::get<1>(result);
                const auto& status=std::get<2>(result);
                require(out.sizes()==at::IntArrayRef({h,512}),"candidate output shape");
                require(scale.sizes()==at::IntArrayRef({h,16}),"candidate scale shape");
                require(out.scalar_type()==at::kByte && scale.scalar_type()==at::kByte,"candidate output dtypes");
                require(out.strides()==at::IntArrayRef({544,1}) && scale.strides()==out.strides(),"packed strides");
                require(out.storage_offset()==0 && scale.storage_offset()==512,"packed byte offsets");
                require(out.is_alias_of(scale),"outputs must view the same 544-byte rows");
                require(!out.is_alias_of(status),"status must be independent");
                require(status.numel()==1 && status.scalar_type()==at::kInt,"status contract");
            }
            for(int h:{8,16,32,64}){
                auto result=base.call(at::empty({1,h,576},byte.dtype(at::kBFloat16)),
                    at::empty({32,256,1,656},byte),idx,table,cuq,len,0.04);
                require(std::get<0>(result).sizes()==at::IntArrayRef({1,h,512}),"baseline shape");
                require(std::get<0>(result).scalar_type()==at::kBFloat16,"baseline dtype");
                require(std::get<1>(result).numel()==0,"baseline has no output scales");
            }
        } else if(mode=="invalid") {
            auto q=at::empty({8,608},byte);
            rejects([&]{low.call(q,at::empty({65536,256,1,416},byte),idx,
                at::empty({1,65536},integer),cuq,len,0.04);});
            rejects([&]{low.call(at::empty({64,608},byte),cache,idx,table,cuq,len,0.04);});
            rejects([&]{low.call(at::empty({8,594},byte),cache,idx,table,cuq,len,0.04);});
            rejects([&]{low.call(q.to(at::kBFloat16),cache,idx,table,cuq,len,0.04);});
            rejects([&]{low.call(q,at::empty({32,256,1,656},byte),idx,table,cuq,len,0.04);});
            rejects([&]{low.call(q,cache,at::empty({1,1,64},integer),table,cuq,len,0.04);});
            rejects([&]{low.call(q,cache,at::empty({1,1,16384},integer),table,cuq,len,0.04);});
            rejects([&]{low.call(q,cache,idx,at::empty({1,31},integer),cuq,len,0.04);});
            rejects([&]{low.call(q,cache,idx,table,cuq.to(at::kLong),len,0.04);});
            rejects([&]{low.call(q,cache,idx,table,cuq,len,0);});
            rejects([&]{low.call(q,cache,idx,table,cuq,len,std::numeric_limits<double>::quiet_NaN());});
            rejects([&]{low.call(q,cache,idx,table,cuq,len,std::numeric_limits<double>::max());});
            rejects([&]{low.call(q,cache,idx,table,cuq,len,std::numeric_limits<double>::min());});
            rejects([&]{low.call(at::empty({9,608},byte).narrow(0,1,8),cache,idx,table,cuq,len,0.04);});
            rejects([&]{low.call(at::empty_strided({8,608},{609,1},byte),cache,idx,table,cuq,len,0.04);});
        } else {throw std::runtime_error("unknown mode");}
        std::cout<<mode<<" passed: actual binding host compile + schemas/Meta only\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
"""


@pytest.fixture(scope="module")
def actual_binding(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("official_binding")
    for relative, content in SHIMS.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source, executable = directory / "meta.cpp", directory / "meta"
    source.write_text(CPP)
    command = [
        compiler,
        "-std=c++17",
        "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I",
        str(directory),
        "-I",
        str(SOURCE.parent),
    ]
    for path in include_paths():
        command.extend(("-I", path))
    command.extend((str(source), "-o", str(executable)))
    for path in library_paths():
        command.extend(("-L", path, f"-Wl,-rpath,{path}"))
    command.extend(("-ltorch", "-ltorch_cpu", "-lc10"))
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return executable


@pytest.mark.parametrize("mode", ["valid", "invalid"])
def test_actual_official_binding_compiles_and_dispatches_meta(actual_binding, mode):
    result = subprocess.run([str(actual_binding), mode], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "actual binding host compile + schemas/Meta only" in result.stdout
