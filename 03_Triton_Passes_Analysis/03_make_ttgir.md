## make_ttgir

> [!NOTE]  
> TTGIR: 추상적인 텐서에 실제 GPU 스레드 배치 정보인 레이아웃을 주입
>
> - 메모리 접근을 효율적으로 할 수 있도록 레이아웃을 세팅
> - 이후 레이아웃 변환 오버헤드를 최대한 줄이는 방향으로 변환
> - `LICM`, `CSE` 등 기존의 최적화 방법들도 GPU 특성에 맞게 확장
> - (PASS 개수가 많기 때문에, 예시 커널에 실제 적용된 PASS만 분석)

### add_convert_to_ttgpuir (TTIR)

- 1~3번 모두에서 변화
- `lib/Conversion/TritonToTritonGPU/TritonToTritonGPU.cpp`에 위치
- 설계 구조는 `Dialect Conversion`과 같음(2장 참고)
  - `populateXXX` 함수들로 `tensor`를 사용하는 모든 연산을 `patterns`에 추가
  - `typeConverter`에서, 텐서에 HW 배치 정보(레이아웃) 추가
    - `tensor` 타입 (MLIR 기본) 내부의 `encoding`이라는 Attribute 활용
  - `typeConverter`에서, 연산 간에 기대하는 데이터 배치가 다를 경우 `ttg.convert_layout` 연산 추가
    - 나중에(add_reduce_data_duplication) HW 타겟에 따라 Shared Memory를 적절히 사용하도록 변환

```
# Before
tensor<128x64xi32>
tensor<32x1x!tt.ptr<i8>>
tt.broadcast %124 : tensor<1x32xi1> -> tensor<128x32xi1>


# After
tensor<128x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>>
tensor<128x32x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>>
ttg.convert_layout %114 : tensor<128x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>> -> tensor<128x!tt.ptr<i8>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>}>>
```

- HW 파라미터를 모듈 전체의 Attribute로 추가
  - module **attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32}**

### add_coalesce (TTGIR)

- 1~3번 모두에서 변화
- `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`에 위치
- `tensor<...x!tt.ptr<>>`를 사용하는 `tt.load` 및 `tt.store` 대상
- `load`/`store` 지점의 레이아웃을 강제로 바꾸기 때문에, 연산 앞뒤로 `ttg.convert_layout`이 삽입

```
# Before
%175 = tt.load %164, %174, %cst : tensor<32x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>>

# After (이미 최적의 레이아웃이므로, 후처리 X)
%181 = ttg.convert_layout %170 : tensor<32x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>> -> tensor<32x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>>
%182 = ttg.convert_layout %180 : tensor<32x64xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>> -> tensor<32x64xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>>
%183 = ttg.convert_layout %cst : tensor<32x64xi8, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>> -> tensor<32x64xi8, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>>
%184 = tt.load %181, %182, %183 : tensor<32x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>>
```

- `buildCoalescedEncoding`에서, `contiguity`가 가장 높은 축을 order[0]으로 설정하여 워프 내 스레드들이 이 방향으로 나란히 서게 함
  - `contiguity`: 단순히 차원 크기가 아니라, 실제 주소값이 1씩 증가하는 최대 연속 길이
- `buildCoalescedEncoding`에서, `contiguity`와 `divisibility`를 기준으로 벡터화가 가능할지 판단
  - `divisibility`: 시작 주소가 몇 바이트 단위(2^n)로 정렬되어 있는지
  - min(Alignment, Contiguity, 128/elemNumBits)
    - 이 값이 4이면 vectorized 명령어 활용
- **TODO**: MSDA에서 Vectorized Memory Load를 하지 않은 이유 디버깅

### add_remove_layout_conversions (TTGIR)

- 1~3번 모두에서 변화
- 불필요한 `convert_layout` 제거 (ex. `add_coalesce`에서 생성된 동일 레이아웃끼리 변환)
- Coalescing에 최적화된 blocked 레이아웃과 텐서 코어 연산에 최적화된 mma 레이아웃 간의 `convert_layout` 오버헤드를 줄이기
  - `convert_layout`은 스레드들이 들고 있는 데이터를 서로 교환해야 하므로 Shared Memory를 거쳐야만 함
  - 1. `Forward`: `tt.load`나 `tt.dot` 등 레이아웃이 고정되어야 하는 기준 연산(Anchor)의 레이아웃을 후속 연산들에 전파하여, 동일한 레이아웃을 그대로 쓰도록 강제
  - 2. `Backward`: `convert_layout`이 생길 경우, 이전 연산을 되짚어가며 원래부터 해당 레이아웃으로 계산할 수 있는지 확인하여 변환을 제거
  - 3. `Hoist`: 지우지 못한 `convert_layout`은 연산 그래프의 최대한 위쪽(`tt.load` 직후 등)으로 올려, 후속 Element-wise 연산들을 이미 변환된 레지스터 위에서 지연 없이 수행

### add_accelerate_matmul (TTGIR)

- 2번에서 변화
- `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`에 위치
- tt.dot의 입/출력 레이아웃을 Tensor Core에 맞게 변환
  - `#ttg.blocked` -> `#ttg.nvidia_mma`
- tt.dot 전후로 `convert_layout`이 생성되었으므로, 이후 `add_remove_layout_conversions` 추가
- **TODO**: PASS 세부 분석

```
# Before
%152 = tt.dot %150, %151, %arg28, inputPrecision = tf32 : tensor<128x32xi8, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>}>> * tensor<32x64xi8, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>}>> -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>>

# After
%152 = ttg.convert_layout %arg28 : tensor<128x64xi32, #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>> -> tensor<128x64xi32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>>
%153 = ttg.convert_layout %150 : tensor<128x32xi8, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>}>> -> tensor<128x32xi8, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>, kWidth = 4}>>
%154 = ttg.convert_layout %151 : tensor<32x64xi8, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>}>> -> tensor<32x64xi8, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>, kWidth = 4}>>
%155 = tt.dot %153, %154, %152, inputPrecision = tf32 : tensor<128x32xi8, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>, kWidth = 4}>> * tensor<32x64xi8, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>, kWidth = 4}>> -> tensor<128x64xi32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>>
%156 = ttg.convert_layout %155 : tensor<128x64xi32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>> -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>>
```

### add_triton_licm (TTIR)

- 2번에서 변화
- `MLIR`의 `LICM` + offset 계산/상수 연산 등도 Hoisting
  - `LICM`(`Loop Invariant Code Motion`): 루프 내에서 불변하는 연산을 밖으로 이동
  - GPU의 `masking` 개념을 활용하여, 더욱 공격적인 LICM 구현
  - `MLIR`의 `moveLoopInvariantCode` 함수에 Lambda 추가하면 됨

### add_loop_aware_cse (TTIR)

- 2, 3번에서 변화
- 일반 `CSE`는 단순히 "똑같은 연산이 두 번 나오면 하나를 지우는" 방식이므로, 아래와 같은 경우를 최적화하지 못함

```python
# 초기값
arg1 = 10
arg2 = 10

for i in range(100):
    # 루프 내부
    val1 = arg1 + 5
    val2 = arg2 + 5
    ...
    arg1 = val1  # 다음 바퀴 준비
    arg2 = val2
```

- 수학적 귀납법을 사용하여 중복된 루프 변수 자체를 제거
  - "루프 시작 시 `A`와 `B`가 같고, 루프 한 바퀴를 돌고 난 후의 `A_next`와 `B_next`도 같다면, `A`와 `B`는 루프 내내 같다"
- 모든 중복 변수가 없어질 때까지 반복 수행
  - 인자가 통합되면서 루프 내부의 공통 부분식이 새롭게 발견될 수 있음

### add_reduce_data_duplication (TTGIR)

- 2번에서만 변화
- 아래 조건을 만족하지 못하는 복잡한 `dot` 연산일 때, shared memory를 사용하도록 변환
  - `cvtReordersRegisters`: 출력 차원이 register뿐이거나 비어있을 때
  - `cvtNeedsWarpShuffle`: 출력 차원이 register와 lane으로 구성되어 있고, 데이터의 Transposition 구조가 단순할 때
- 데이터 재사용성을 높일 수 있고, 같은 변수(matrix)를 여러 번 사용한다면 `ttg.local_load`는 CSE 적용 가능

```
# Before
%147 = ttg.convert_layout %137 : tensor<128x32xi8, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>> -> tensor<128x32xi8, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>, kWidth = 4}>>

# After
%147 = ttg.local_alloc %137 : (tensor<128x32xi8, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x32xi8, #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [1, 0]}>, #ttg.shared_memory>
%148 = ttg.local_load %147 : !ttg.memdesc<128x32xi8, #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [1, 0]}>, #ttg.shared_memory> -> tensor<128x32xi8, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>, kWidth = 4}>>
```

### add_reorder_instructions (TTGIR)

- 2번에서만 변화
- 명령어의 실행 순서를 재배치하여 레지스터 압박을 줄임
  - `ttg.convert_layout`은 공유 메모리 해제 직후나, 이 값이 실제로 처음 사용되기 전으로 밀어버림
  - `ttg.local_alloc`은 소스 데이터가 나오자마자(`tt.load` 이후) 함
  - 위 연산들이 레지스터 압박을 높일 경우, 루프 내부로 옮김

### add_sccp (MLIR)

- 2, 3번에서 변화
- `SCCP`(`Sparse Conditional Constant Propagation`): 실행해 보지 않아도 알 수 있는 값은 미리 계산 (2장 참고)
