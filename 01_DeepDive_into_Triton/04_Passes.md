## Brief about Passes

- Triton에서 적용되는 많은 Pass들이 어떤 구조로 정의되어 있는가를 간단하게 살펴봄
- 자세한 내용은 `02_MLIR_Tutorial`을 참고

### TableGen

- `include/triton/Dialect/TritonGPU/Transforms/Passes.td`
- `TableGen`에서는 각 Pass의 이름과 namespace 등만 지정

```
def TritonGPUCoalesce: Pass<
  "tritongpu-coalesce",   // 커맨드 라인에서 부를 이름 (ex: triton-opt --tritongpu-coalesce)
  "mlir::ModuleOp"> {
  let summary = "coalesce";

  let description = [{
    The pass analyses loads/stores with type `tensor<tt.ptr<>>` or
    `tt.ptr<tensor<>>` and replaces the layouts of these operations with
    coalesced layouts, i.e. cache friendly access patterns.
    Layout conversions are inserted before and after the load/store op
    to maintain consistency with the rest of the program.
  }];

  // 의존성 (이 Pass를 위해 필요한 Dialect들)
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}
```

### C++

- `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`
- `runOnOperation()`에서 실제 Pass 함수 구현

```C++
void runOnOperation() override {
  // Run axis info analysis
  ModuleOp moduleOp = getOperation();
  ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  ...

  // For each memory op that has a layout L1:
  // 1. Create a coalesced memory layout L2 of the pointer operands
  // 2. Convert all operands from layout L1 to layout L2
  // 3. Create a new memory op that consumes these operands and
  //    produces a tensor with layout L2
  // 4. Convert the output of this new memory op back to L1
  // 5. Replace all the uses of the original memory op by the new one
  for (auto &kv : layoutMap) {
    convertDistributedOpEncoding(kv.second, kv.first);
  }
}
```

### Build

- `setup.py`를 실행하면 Triton의 C++ 코드와 함께 LLVM/MLIR도 빌드
- 이 과정에서
  `include/triton/Dialect/TritonGPU/Transforms/CMakeLists.txt`에 정의된 설정에 따라
  MLIR TableGen이 실행
- TableGen은 TritonGPU dialect에 정의된 pass들을 기반으로 `Passes.h.inc` 파일을 자동 생성
- 생성된 `Passes.h.inc`는 `Passes.h`에서 include
- 이후 `passes.cc`에서 `Passes.h`에 선언된 Pass 생성 함수들을 pybind11으로 래핑
- 그 결과, 해당 Pass들은 Python에서 아래처럼 import하여 사용할 수 있는 shared library로 빌드됨

```python
from triton._C.libtriton import passes
```
