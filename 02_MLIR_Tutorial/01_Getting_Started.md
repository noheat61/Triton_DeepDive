## [MLIR — Getting Started](https://www.jeremykun.com/2023/08/10/mlir-getting-started/)

> [!IMPORTANT]  
> 예제 코드의 호환성을 위해 **Bazel** 사용을 권장  
> CMake도 가능하나 설정이 번거로울 수 있음

### 1. Bazel 설치

설치 가이드: [Bazel Install Guide](https://bazel.build/versions/9.0.0/install/ubuntu?hl=ko)

### 2. 프로젝트 빌드

```shell
bazel build ...:all
bazel test ...:all
```

> 💡 왜 MLIR은 LLVM 위에서 개발되었나?  
> MLIR 프로젝트를 리딩한 Chris Lattner가 바로 LLVM의 창시자  
> LLVM IR만으로는 다루기 어려웠던 문제를 해결하기 위해, LLVM 인프라의 장점을 활용하되 더 유연한 MLIR을 설계

## [Running and Testing a Lowering](https://www.jeremykun.com/2023/08/10/mlir-running-and-testing-a-lowering/)

MLIR의 핵심 메커니즘인 **Lowering**을 직접 수행하며 감을 잡는 단계  
원문에서는 lit, FileCheck 같은 테스트 도구도 다루지만, 현재 단계에서는 MLIR의 동작 원리 이해에 집중하기 위해 생략

> [!WARNING]  
> bazel run 실행 시 시간이 오래 걸릴 수 있습니다.  
> Bazel이 최신 변경 사항을 감지하여 자동으로 재빌드 후 실행하는 과정이니 기다려주세요.

```shell
# Step 1. 기본 실행
bazel run @llvm-project//mlir:mlir-opt -- $(pwd)/tests/ctlz.mlir

# Step 2. 단일 패스 적용
bazel run @llvm-project//mlir:mlir-opt -- $(pwd)/tests/ctlz.mlir --convert-math-to-funcs=convert-ctlz

# Step 3. 복합 파이프라인 적용
bazel run @llvm-project//mlir:mlir-opt -- $(pwd)/tests/ctlz_runner.mlir \
 --pass-pipeline="builtin.module( \
 convert-math-to-funcs{convert-ctlz}, \
 func.func(convert-scf-to-cf, convert-arith-to-llvm), \
 convert-func-to-llvm, \
 convert-cf-to-llvm, \
 reconcile-unrealized-casts)"
```
