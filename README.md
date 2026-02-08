# DeepDive into Triton/MLIR

- OpenAI의 **Triton** 언어와 그 기반이 되는 **MLIR(Multi-Level Intermediate Representation)** 인프라의 내부 동작 원리를 심층 분석
- Triton이 Python 코드를 고성능 GPU 커널로 변환하는 과정에서 MLIR과 어떻게 연계되는지 살펴보고, 구체적인 CUDA 커널 최적화 메커니즘을 파악

## Triton이란 무엇인가?

Triton은 OpenAI에서 개발한 GPU 프로그래밍 언어 및 컴파일러입니다.  
CUDA와 달리 Block 단위의 프로그래밍 모델을 제공하여 개발 생산성을 높이면서도, 컴파일러 레벨의 최적화를 통해 CUDA 라이브러리와 대등한 성능을 낼 수 있습니다.

- **GitHub**: [triton-lang/triton](https://github.com/triton-lang/triton)
- **Reference**: [OpenAI Triton Keynote (YouTube)](https://www.youtube.com/watch?v=AtbnRIzpwho)
- **Reference**: [Triton Introduction (YouTube)](https://www.youtube.com/watch?v=fxNud9m1F8I) (by [@triangle](https://www.inflearn.com/users/163955/@triangle))

## MLIR이란 무엇인가?

MLIR은 LLVM 프로젝트의 일환으로, Pytorch compile, Triton 등 다양한 컴파일러를 구축하기 위한 재사용 가능한 인프라입니다.  
Triton은 MLIR의 Dialect 시스템을 활용하여 고수준 최적화와 GPU 하드웨어 매핑을 수행합니다.

- **Intro PDF(by xDSL)**: [Introduction to MLIR and LLVM](https://github.com/xdslproject/training-intro/raw/main/lectures/Introduction%20to%20MLIR%20and%20LLVM.pdf)
- **Official Docs**: [mlir.llvm.org](https://mlir.llvm.org/)

## 📚 목차

1. **Deep Dive into Triton**

- Triton 설치부터 컴파일러 패스(Pass) 디버깅까지 직접 수행하며, 고수준 코드가 GPU 커널로 변환되는 과정과 MLIR의 연동 메커니즘을 심층 분석합니다.
- ref: [Deep Dive into Triton Internals](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/)

2. **MLIR Tutorial & Fundamentals**

- MLIR의 아키텍처를 이해하고, 커스텀 Dialect 정의 및 Lowering 파이프라인 구축 과정을 통해 LLVM 백엔드와의 연계 방식을 학습합니다.
- ref: [MLIR Tutorial](https://github.com/j2kun/mlir-tutorial?tab=readme-ov-file#mlir-tutorial)

3. **Triton Passes Analysis**

- 앞서 학습한 내용을 바탕으로 Triton의 각 최적화 패스(Optimization Pass)가 MLIR 레벨에서 구현되는 방식을 분석합니다.
- 이를 통해 구체적인 CUDA 커널 최적화 알고리즘의 원리를 파악합니다.
