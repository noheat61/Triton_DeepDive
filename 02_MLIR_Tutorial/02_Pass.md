## [Writing Our First Pass](https://www.jeremykun.com/2023/08/10/mlir-writing-our-first-pass/)

MLIR 컴파일러 개발의 핵심은 **Pass**를 정의하는 것  
Pass는 크게 세 가지 역할을 수행

1. **Optimization**: 프로그램 최적화
2. **Lowering**: 다른 Dialect로 변환
3. **Canonicalization**: 정규화 (불필요한 연산 정리 등)

### 프로젝트 구조

사용자 Pass를 추가하여 mlir-opt를 커스터마이징  
튜토리얼 초반에는 편의상 lib에 합쳐져 있으나, 실제로는 아래와 같이 헤더와 구현부가 분리

- `include/`: 헤더 파일(`.h`) 및 TableGen 정의 파일(`.td`)
- `lib/`: 실제 구현 소스 코드(`.cpp`)
  - `Transform`: IR 변환 Pass
  - `Conversion`: Dialect 간 Lowering Pass
  - `Analysis`: 분석 Pass (수정 없이 정보만 수집)
- `mlir-opt.cpp`: 작성한 Pass를 등록(Registry)하고 실행하는 진입점

### Dialect 순회

가장 직관적인 방법은 IR 트리를 직접 순회하며 원하는 패턴을 찾는 것

#### 1. Pass 클래스 선언 (`AffineFullUnroll.h`)

```C++
class AffineFullUnrollPass
: public PassWrapper<AffineFullUnrollPass, OperationPass<mlir::func::FuncOp>>
```

- `PassWrapper`: CLI 인자 파싱, --help 출력 등 번거로운 작업을 대신 처리
- `AffineFullUnrollPass`: [CRTP](https://wikidocs.net/501), 자기 자신을 템플릿 인자로 넘김
- `OperationPass<mlir::func::FuncOp>`: 이 Pass의 실행 단위를 FuncOp(함수)로 제한
  - MLIR은 병렬 처리를 수행하므로, 서로 독립된 영역을 명시해야 안전하게 최적화할 수 있음
  - getOperation 메서드도 OperationPass에서 제공

주요 오버라이딩 함수:

- `runOnOperation`: 실제 Pass 로직이 수행되는 메인 함수
- `getArgument`: CLI에서 이 Pass를 호출할 때 쓸 플래그 이름
- `getDescription`: --help 실행 시 보여줄 설명

#### 2. Pass 구현 (`AffineFullUnroll.cpp`)

```C++
void AffineFullUnrollPass::runOnOperation() {
  getOperation().walk([&](AffineForOp op) {
    if (failed(loopUnrollFull(op))) {
      op.emitError("unrolling failed");
      signalPassFailure();
    }
  });
}
```

- `getOperation().walk()`: 현재 Operation(여기선 함수) 하위의 모든 노드를 순회
  - walk는 함수뿐만 아니라 모듈, 루프 등 모든 Operation이 가진 기본 메서드
- AST를 후위 순회하면서, 콜백 함수의 인자 타입(AffineForOp)을 만나면 함수 실행
- `loopUnrollFull`: 실제 변환을 수행하는 함수
  - 이번 예제에서는 MLIR 내장 함수를 빌려 씀

#### 3. 테스트

```shell
bazel run //tools:tutorial-opt -- $(pwd)/tests/affine_loop_unroll.mlir -affine-full-unroll
```

### Pattern Rewrite

단순 walk 방식은 복잡한 변환 로직을 짜거나, 변환 실패 시 Rollback 처리하기 어려움  
MLIR은 이를 위해 Pattern Rewrite System을 제공

- 안전성: 변환 도중 실패(failure)하면 자동으로 변경 사항을 되돌려줌
- 유연성: walk보다 훨씬 정교한 순회 및 반복 적용이 가능
- applyPatternsAndFoldGreedily를 쓰면 x \* 1 → x 같은 자잘한 최적화도 자동으로 적용

#### 1. 패턴 정의 (`OpRewritePattern`)

```C++
struct AffineFullUnrollPattern :
  public OpRewritePattern<AffineForOp> {
  AffineFullUnrollPattern(mlir::MLIRContext *context)
      : OpRewritePattern<AffineForOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    return loopUnrollFull(op);
  }
};
```

- `OpRewritePattern<TargetOp>`: 특정 Operation(TargetOp)을 찾아서 어떻게 바꿀지 정의하는 클래스
- `matchAndRewrite`: 패턴이 일치하는지 확인(match)하고, 일치하면 변환(rewrite)을 수행
- `PatternRewriter`: IR을 수정할 때 사용하는 도구
  - `PatternRewriter`을 사용해 `matchAndRewrite`를 구현하는 것이 일반적이지만, 여기서는 MLIR의 `loopUnrollFull` 내장 함수 사용
  - 튜토리얼에 곱셈의 case는 잘 구현되어 있으므로 참고

#### 2. 패턴 등록 및 실행

```C++
struct AffineFullUnrollPatternRewrite
    : impl::AffineFullUnrollPatternRewriteBase<AffineFullUnrollPatternRewrite> {
  using AffineFullUnrollPatternRewriteBase::AffineFullUnrollPatternRewriteBase;
  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AffineFullUnrollPattern>(&getContext());
    // One could use GreedyRewriteConfig here to slightly tweak the behavior of
    // the pattern application.
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
```

- `RewritePatternSet`: 내가 정의한 패턴들을 담는 바구니
- `applyPatternsAndFoldGreedily`: 등록된 패턴들을 적용 가능한 곳이 없을 때까지 반복적으로 실행

## [Using Tablegen for Passes](https://www.jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/)

MLIR은 C++로 작성해야 할 반복적인 코드를 줄이기 위해 `TableGen`을 사용  
Pass뿐만 아니라 Dialect, Operation 등 MLIR 생태계 전반에서 쓰임

#### 1. Pass 정의(`Passes.td`)

```
include "mlir/Pass/PassBase.td"

def AffineFullUnroll : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops";
  let description = [{
    Fully unroll all affine loops. (could add more docs here like code examples)
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

```

- C++ 헤더 대신 DSL(Domain Specific Language) 형태로 Pass 정보를 정의
  - `def` (`AffineFullUnroll`): 실제 Pass 정의
  - `class`: 상속 및 재사용을 위한 템플릿 (`def`와 달리, 빌드되지 않음)
  - `let`: 옵션, 설명 등 메타데이터 설정

#### 2. 빌드 과정 (`lib/Transform/Affine/BUILD`)

1.  `mlir-tblgen` 도구가 `Passes.td` 파일을 읽음
2.  `Passes.h.inc`와 같은 C++ 헤더 조각을 생성
3.  개발자는 `.cpp` 파일에서 `#include "Passes.h.inc"`로 불러와서 사용
4.  핵심 로직인 `runOnOperation()`은 직접 C++로 구현
