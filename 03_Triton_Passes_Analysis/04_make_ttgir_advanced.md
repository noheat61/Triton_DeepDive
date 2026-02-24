## make_ttgir (advanced)

> [!NOTE]  
> 앞서 언급되지 않았던 PASS들도 간단히 분석

### 연산 및 레이아웃 최적화

- `add_f32_dot_tc`
  - 고정밀도(F32) 행렬 곱셈을 하드웨어가 더 잘하는 저정밀도 엔진(TF32, BF16, FP16)의 조합으로 분해
  - 정밀도 손실을 최소화하면서 Tensor Core를 강제로 가속
- `nvidia.passes.ttnvgpuir.add_plan_cta`
  - Hopper(SM90) 아키텍처의 핵심인 CTA Cluster와 Distributed Shared Memory(DSMEM) 활용 계획을 수립
  - 개별 CTA 단위를 넘어 클러스터 단위의 멀티캐스트 및 데이터 공유 전략을 설계
- `add_optimize_thread_locality`
  - Reduction이나 Gather 연산 시 발생하는 스레드 간 동기화 비용을 감소
- `add_optimize_dot_operands`
  - 행렬 곱셈 피연산자의 레이아웃을 하드웨어 가속기(`MMA`)가 즉시 소비할 수 있는 형태로 재배치
  - 전치 연산을 하드웨어 로딩 명령어 내부로 Fuse하고, 뱅크 충돌 방지를 위한 Swizzling 및 양자화용 Scale 관리 전략

### 제어문 및 루프 정리

- `add_fuse_nested_loops`
  - 중첩된 루프를 하나의 거대한 루프로 합쳐 반복문 관리 오버헤드를 줄임
- `add_combine_tensor_select_and_if`
  - 같은 조건을 쓰는 select와 if 문을 하나로 합쳐서 중복 계산을 줄임

### 스케줄링 및 파이프라이닝

- `add_assign_latencies`
  - pipeline 전에 흥미로운 연산들에 대한 레이턴시 할당
  - 절대적인 소요 시간이 아닌, 결과를 얻기 위해 대기해야 하는 스테이지 수(`numStages`)
- `add_schedule_loops`
  - 레이턴시를 기반으로 루프 안의 명령어들의 순서를 재배치하여, 데이터를 기다리느라 발생하는 코어 유휴 시간을 최소화
- `add_pipeline`
  - 비동기 메모리 로드와 연산이 멈춤 없이 겹쳐서 수행되는 파이프라인 구조를 구축

### 최신 하드웨어 특화 아키텍처 (SM90/SM100)

- Warp Specialization (`nvidia.passes.hopper.add_hopper_warpspec` / `add_warp_specialize` / `add_optimize_partition_warps`)
  - `Warp`들에게 동일한 작업을 지시하지 않고, 메모리를 옮기는 역할(`Producer`)과 계산하는 역할(`Consumer`)로 분담
  - SM90 이상에서 워프 간의 비동기적 협력을 유도하여 하드웨어 파이프라인 성능을 극대화
- Tensor Memory (`add_hoist_tmem_alloc` / `add_promote_lhs_to_tmem` / `add_remove_tmem_tokens` / `add_optimize_tmem_layouts` / `add_interleave_tmem`)
  - `Blackwell`(SM100) 아키텍처의 핵심인 `Tensor Memory`(`TMEM`) 활용을 위한 전반적인 최적화 수행
  - TMEM 할당을 루프 외부로 빼내고, 행렬 A(Lhs)를 TMEM으로 올리며, 동기화 토큰 정리 및 레이아웃을 재배치
- Tensor Memory Accelerator (TMA): `nvidia.passes.ttnvgpuir.add_tma_lowering`
  - 범용적인 Load/Store 연산들을 SM90 이상 최신 GPU의 TMA 전용 하드웨어 명령어로 Lowering
- Tensor Memory Accelerator (TMA): `nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding`
  - TMA를 위한 전용 Descriptor을 최적화
  - 디스크립터의 인코딩을 최적화하여 글로벌 메모리와 공유 메모리 간의 Zero-overhead 데이터 이동을 실현

- MMA Lowering (`nvidia.passes.ttnvgpuir.add_lower_mma`)
  - SM90 이상에서 행렬 곱셈 연산을 실제 Tensor Core를 직접 구동하는 저수준 어셈블리 명령어(예: wgmma.mma_async, mma.sync 등)로 변환

### 기타 메모리 최적화

- `add_prefetch`
  - 메모리를 캐시로 미리 가져와(Prefetch) 데이터 로딩에 따른 지연 시간을 은닉
- `add_coalesce_async_copy`
  - 잘게 쪼개진 여러 번의 비동기 복사(`cp.async`) 요청을 큰 덩어리로 병합하여 메모리 대역폭 낭비를 방지
