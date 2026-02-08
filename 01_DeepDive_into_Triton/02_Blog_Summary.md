## Summary of Blog Post

- 아래 블로그 포스트를 베이스로, Triton 컴파일러의 내부 동작 원리를 코드 레벨에서 분석
- link: [BLOG](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/)

### 1. Triton Compiler Architecture

#### Triton 컴파일 패스 (vs NVCC)

- NVCC(CUDA Compiler)는 소스 코드를 입력받아 전처리 후 IR을 생성하는 과정이 **Closed-source**
- Triton은 LLVM과 MLIR을 기반으로 한 **Open-source** 컴파일러
  - Python AST에서 출발하여, 목적에 맞는 Dialect를 거치며 구체화
  - 표준 LLVM IR로 변환되며, 여기서 LLVM의 최적화 패스들을 적용
  - LLVM NVPTX 백엔드를 통해 PTX 코드를 생성
  - **Note**: SASS로 변환할 때는 NVIDIA Driver(JIT Compiler) / ptxas를 활용, **Closed-source**

|                                 NVCC(CUDA)                                  |                                                                                             Triton                                                                                             |
| :-------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://www.kapilsharma.dev/assets/cuda_compile.jpg" width="800"> | <img src="https://www.kapilsharma.dev/assets/triton-compiler-middle-layer.png" width="800"><br><br><img src="https://www.kapilsharma.dev/assets/triton-compiler-pytorch-conf.png" width="800"> |

#### IR Pipeline: Python to Binary

| 단계 | IR 이름                   | 추상화 수준 | 주요 특징                                                                                       |
| ---- | ------------------------- | ----------- | ----------------------------------------------------------------------------------------------- |
| 1    | **TTIR (Triton IR)**      | 매우 높음   | 논리적 연산과 데이터 흐름만 정의<br>하드웨어 세부 구현 사항 없음                                |
| 2    | **TTGIR (Triton GPU IR)** | 중간        | 논리적 연산을 GPU 실행 모델에 매핑<br>계층 구조, 메모리 레이아웃 등 타겟 하드웨어 스펙 반영     |
| 3    | **LLIR (LLVM IR)**        | 매우 낮음   | NVIDIA 백엔드에 완전히 종속<br>`llvm.nvvm.read.ptx.sreg.tid.x()` 등 하드웨어 레지스터 직접 접근 |
| 4    | **PTX**                   |             | CUDA용 저수준 가상 머신 및 어셈블리 언어<br> 특정 GPU 아키텍처에 종속되지 않음                  |
| 5    | **CUBIN**                 |             | 실제 HW가 실행하는 기계어<br> 대상 아키텍처에 최적화된 바이너리                                 |

#### Code Example: Vector Add

```python
import torch

import triton
import triton.language as tl
from triton.tools.disasm import get_sass

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

size = 1024
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

output = torch.empty_like(x)
compiled_kernel = add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)

print(compiled_kernel.asm.keys())
# print(compiled_kernel.asm["ttir"])
# print(compiled_kernel.asm["ttgir"])
# print(compiled_kernel.asm["llir"])
# print(compiled_kernel.asm["ptx"])
# print(get_sass(compiled_kernel.asm["cubin"]))
```

### 2. Compiler Frontend

#### CLI command

```shell
python3 python/triton/tools/compile.py \
  --kernel-name add_kernel \
  --signature "*fp32,*fp32,*fp32,i32,64" \
  --grid=1024,1024,1024 \
  vector-add.py
```

#### Entry Point: [python/triton/tools/compile.py](https://github.com/triton-lang/triton/blob/main/python/triton/tools/compile.py#L80-L207)

- Python 소스 파싱 및 ASTSource 객체 생성
- ASTSource에는 커널 코드, 상수, 속성 정보가 포함
- `triton.compile()`을 호출하여 본격적인 컴파일을 시작

#### Specialization: [python/triton/compiler/compiler.py](https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py#L226-L363)

- `add_stages`: 타겟 백엔드에 맞는 최적화 단계(Passes)를 설정
- `make_ir`: AST를 초기 MLIR(TTIR)로 변환
- `compile_ir`: 정의된 단계별로 IR 최적화 및 변환
- `backend` 변수는 빌드 시 third_party/nvidia로부터 symbolic link 연결

### 3. Compiler Backend

#### Add Stages: [third_party/nvidia/backends/nvidia/compiler.py](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py#L545-L556)

- Dialect(TTIR, TTGIR 등) 변환에 필요한 Pass들을 파이프라인에 추가

#### Make TTIR: [third_party/nvidia/backends/nvidia/compiler.py](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py#L234-L249)

- `ir.pass_manager`: MLIR의 PassManager 객체 (C++ 바인딩)
- `passes.common`, `passes.ttir` 등 다양한 Dialect에서 최적화 패스 등록

### 4. MLIR Passes 상세

#### Code Example: MLIR Passes (C++)

```C++
mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get());
mlir::PassManager pm(module);

// Add some passes
pm.addPass(mlir::createCSEPass());
pm.addPass(mlir::createDeadCodeEliminationPass());

pm.run(module);
```

#### Common Passes (passes.common): LLVM/MLIR 표준 패스들을 래핑

- ex. `passes.common.add_inliner` -> createInlinerPass (llvm-project 정의)

#### TTIR Passes (passes.ttir): Triton 관련, triton/lib/Dialect/Triton에 정의

- ex. `passes.ttir.add_rewrite_tensor_pointer` -> TritonRewriteTensorPointer

#### TritonGPU Passes (passes.ttgpuir): **GPU 최적화**, triton/lib/Dialect/TritonGPU에 정의

- Coalescing
- F32 Dot Product Optimization
- CTA Planning / Thread Locality
- Layout Conversion & Shared Memory Allocation
- TMA (Tensor Memory Accelerator) Lowering
