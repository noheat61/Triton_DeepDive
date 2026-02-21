## make_ttir

> [!NOTE]  
> 하드웨어 독립적으로, Triton 언어 수준에서 코드를 최대한 효율적으로 재조합
>
> - `tl.make_block_ptr` 같은 기능을 하드웨어가 계산할 수 있는 주소 연산 수식으로 변환
> - `Inliner`, `CSE`, `Canonicalizer` 등을 수행하여 가벼운 상태로 IR 수정

### add_inliner (MLIR)

1~3번에서 모두 변화  
`llvm-project/mlir/lib/Transforms/InlinerPass.cpp`에 위치  
64비트 정규화 과정에서 발생하는 불필요한 캐스팅이나 중복 코드를 제거

### add_rewrite_tensor_pointer (TTIR)

3번에서 변화  
`include/triton/Dialect/Triton/Transforms/Passes.td`에 위치  
`tt.make_tensor_ptr`와 `tt.advance`를 제거하고 기본 포인터 연산으로 변경

- Triton 커널을 작성할 때, N차원 데이터를 낱개가 아닌 블록(Tile) 단위로 빠르게 읽기 위해 `tl.make_block_ptr`을 사용
- 컴파일러의 IR 단계에서 `tt.make_tensor_ptr`(텐서 포인터 생성)과 `tt.advance`(다음 블록으로 이동)로 번역
- 숨겨져 있던 복잡한 인덱스 및 오프셋 계산이 실제 기본 포인터 연산으로 풀림

복잡한 IR 변환 로직이 사용되므로, `PatternRewriter` 대신 루프 순회

### add_rewrite_tensor_descriptor_to_pointer (TTIR)

최신 GPU(`Hopper`, SM90 이상)에서 사용하는 TMA 전용 디스크립터를 동일한 연산을 수행하는 일반 포인터 명령어로 되돌림

- Triton 커널을 작성할 때, `tl.make_tensor_descriptor` 함수를 직접 호출하여 TMA를 사용할 수 있음
- TMA: 글로벌 메모리와 공유 메모리 간에 대용량 데이터를 효율적으로 전송할 수 있는 가속기

### add_canonicalizer (MLIR)

3번에서 변화  
코드를 '표준 형태'로 변환 (2장 참고)

### add_combine (TTIR)

2, 3번에서 변화

1. Fused Dot (FMA): `(a \* b) + c`를 `dot(a, b, c)`로 합쳐 메모리 접근을 줄임
2. Pointer Folding: 여러 번의 `addptr`을 하나로 합쳐 주소 계산 오버헤드를 줄임
3. Masked Load: `select` 조건문을 사용하는 대신 하드웨어의 `load` 마스킹 기능을 쓰도록 변환
4. reshaped_constant: 상수를 `broadcast`하는 연산 대신, 미리 그 모양을 가진 상수 데이터로 변환
5. Tensor Core Mapping: `dot`으로 변환 가능한 `sum` 연산을 `dot`으로 변환

- `torch.sum(x[:,:,None].expand(-1,-1,n) * y[None,:,:].expand(m,-1,-1),1)`
  => `dot(x,y,splat(0))`

### add_reorder_broadcast (TTIR)

3번에서 변화  
`broadcast(a) + broadcast(b)` -> `broadcast(a + b)`

### add_cse (MLIR)

2, 3번에서 변화  
`CSE`(`Common Subexpression Elimination`): 똑같은 계산이 두 번 나오면 하나만 계산하고 그 결과를 재사용

### add_symbol_dce (MLIR)

`DCE`(`Dead Code Elimination`): 결과가 프로그램의 최종 출력에 아무런 영향을 주지 않는 코드를 찾아 삭제

### add_loop_unroll (TTIR)

`tt.loop_unroll_factor`가 있을 때 loop unrolling 수행

- Triton 커널을 작성할 때, unroll을 명시적으로 수행할지 정할 수 있음
- ex. `for i in tl.range(0, 128, unroll_factor=4):`
