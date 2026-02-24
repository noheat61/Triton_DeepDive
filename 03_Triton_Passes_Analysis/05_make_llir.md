## make_llir

> [!NOTE]  
> 만들어진 코드를 LLVM으로 Lowering

### 메모리 및 리소스 할당

- `add_allocate_warp_groups`: Hopper 아키텍처 등에서 사용되는 Warp Group 단위의 스케줄링을 위한 리소스를 할당
- `add_allocate_shared_memory_nv`: GPU의 Shared Memory(L1 캐시/SRAM) 공간을 정적으로 할당
- `add_allocate_tensor_memory`: 레지스터나 로컬 메모리에 텐서 데이터를 배치하기 위한 공간을 확보
- `add_check_matmul_two_cta`: Matmul 시 두 개의 CTA(Cluster of Thread Blocks)가 협력하여 연산할 수 있는지 확인하고 최적화
- `add_allocate_global_scratch_memory`: 커널 실행 중 임시 저장이 필요한 대용량 데이터를 위해 Global Memory의 스크래치 공간을 할당

```
# Before
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {

# After
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 6144 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32, "ttng.two-ctas" = false} {
```

### add_scf_to_cf

- 2, 3번에서 변화
- `scf.for`나 `scf.if`를 `cf.br`, `cf.cond_br` 등 Jump 명령어 기반의 Control Flow로 변환 (2장 참고)

### add_inliner (GLUON)

- 여기서 변화되지는 않지만, Gluon의 개념을 알아두면 좋을 듯
- Gluon: 저수준 제어를 통한 타일 기반 GPU 프로그래밍
- **Reference**: [Triton Conference 2025](https://www.youtube.com/watch?v=KqeI23SpJx8&list=PLc_vA1r0qoiQqCdWFDUDqI90oY5EjfGuO&index=8)
- **Reference**: [LinkedIn comment by Jemin Lee](https://www.linkedin.com/posts/jemin-lee-7759aa76_triton-conf-2025-%EC%A0%95%EB%A6%AC-34-3-gluon-%EC%A0%80%EC%88%98%EC%A4%80-activity-7388373063958048768-hMWQ/?originalSubdomain=kr)

## add_to_llvmir

- LLVM IR로 Lowering하는 Dialect Conversion (2장 참고)
- `third_party/nvidia/include/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp` 참고
- **TODO**: TritonGPUToLLVM 세부 분석
