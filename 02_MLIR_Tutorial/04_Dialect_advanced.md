## [Folders and Constant Propagation](https://www.jeremykun.com/2023/09/11/mlir-folders/)

sccp (sparse conditional constant propagation): 연산의 출력값이 상수인지 여부를 추론하고, 해당 연산을 상수 값으로 대체하는 방식
이를 위해 세 가지 요소가 유기적으로 작동

1. `Folding` (폴딩): 연산의 입력이 모두 상수라면, 결과도 상수로 계산해서 합쳐버리는 메커니즘
2. SCCP: Folding을 이용해 상수를 찾고, 그 상수를 다음 연산으로 계속 전달하는 Pass
3. `Materialization`: 계산된 상수 값을 실제 IR 코드로 변환해주는 규칙

> [!NOTE]  
> `Folding`은 매우 보수적입니다. 기존 연산의 결과가 이미 존재하는 값이나 상수로 떨어질 때만 작동하고, `x * 2` → `x + x` 처럼 새로운 연산을 조합해서 만드는 건 못합니다.  
> 이런 복잡한 변환은 다음 단계인 `Canonicalization`에서 수행합니다.

#### 1. 상수 연산 정의

```
def Poly_ConstantOp : Op<Poly_Dialect, "constant", [Pure, ConstantLike]> {
  let summary = "Define a constant polynomial via an attribute.";
  let arguments = (ins AnyIntElementsAttr:$coefficients);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$coefficients attr-dict `:` type($output)";
}
```

- `ConstantLike`: SCCP Pass가 "아, 여기서부터 상수가 시작되는구나"라고 인지하는 깃발
- `AnyIntElementsAttr`: 다항식 계수는 여러 개의 정수이므로 배열 형태의 속성을 사용

#### 2. Folding 구현

연산에 대한 TableGen에서 `let hasFolder = 1;`을 추가하면, C++ 헤더(`.inc`)에 `fold` 메서드 선언이 자동으로 생김  
각 연산마다 `fold` 함수를 오버로딩하여 구현

```C++
// 1. poly.constant
OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
  return adaptor.getCoefficients();
}

// 2. poly.add
OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor) {
  return constFoldBinaryOp<IntegerAttr, APInt>(
      adaptor.getOperands(), [&](APInt a, APInt b) { return a + b; });
}

// 3. poly.mul
OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor) {
  auto lhs = cast<DenseIntElementsAttr>(adaptor.getOperands()[0]);
  auto rhs = cast<DenseIntElementsAttr>(adaptor.getOperands()[1]);
  auto degree = getResult().getType().cast<PolynomialType>().getDegreeBound();
  auto maxIndex = lhs.size() + rhs.size() - 1;

  SmallVector<APInt, 8> result;
  result.reserve(maxIndex);
  for (int i = 0; i < maxIndex; ++i) {
    result.push_back(APInt((*lhs.begin()).getBitWidth(), 0));
  }

  int i = 0;
  for (auto lhsIt = lhs.value_begin<APInt>(); lhsIt != lhs.value_end<APInt>();
       ++lhsIt) {
    int j = 0;
    for (auto rhsIt = rhs.value_begin<APInt>(); rhsIt != rhs.value_end<APInt>();
         ++rhsIt) {
      // index is modulo degree because poly's semantics are defined modulo x^N = 1.
      result[(i + j) % degree] += *rhsIt * (*lhsIt);
      ++j;
    }
    ++i;
  }

  return DenseIntElementsAttr::get(
      RankedTensorType::get(static_cast<int64_t>(result.size()),
                            IntegerType::get(getContext(), 32)),
      result);
}
```

- `FoldAdaptor`: 입력값이 상수인지 아닌지 편하게 확인하게 해주는 도우미 객체
- 덧셈/뺄셈은 `constFoldBinaryOp` 같은 MLIR 내장 도구를 사용할 수 있지만, 다항식 곱셈처럼 복잡한 건 직접 구현

#### 3. Materialization 구현

연산에 대한 TableGen에서 `let hasConstantMaterializer = 1;`을 추가하면, C++ 헤더(`.inc`)에 `Operation *PolyDialect::materializeConstant`가 자동으로 생김  
PolyDialect.cpp에서 구현

```C++
Operation *PolyDialect::materializeConstant(
    OpBuilder &builder, Attribute value, Type type, Location loc) {
  auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
  if (!coeffs)
    return nullptr;
  return builder.create<ConstantOp>(loc, type, coeffs);
}
```

## [Verifiers](https://www.jeremykun.com/2023/09/13/mlir-verifiers/)

MLIR 컴파일러는 각 Pass가 실행되기 전후에 `Verifier`를 실행  
IR이 망가지지 않았는지, 규칙을 위반하지 않았는지 자동으로 검사

#### 1. Trait 기반 Verifier

`PolyOps.td`에서 `SameOperandsAndResultType` Trait을 추가하면, MLIR이 알아서 C++ 검증 코드를 생성

- TableGen이 `.inc` 파일에 `verifyTrait` 함수를 생성
- inferReturnTypes 함수도 생성
  - 어셈블리를 `(type, type) -> type`에서 `type`으로 한 번만 작성하게 간소화
- `build` 함수도 return_state를 추론하도록 수정
  - `builder.create`를 사용할 때 `builder.create<PolyAddOp>(loc, lhs.getType(), lhs, rhs);` -> `builder.create<PolyAddOp>(loc, lhs, rhs);`로 간소화

#### 2. Custom Op Verifier

연산에 대한 TableGen에서 `let hasVerifier = 1;`을 추가하면, C++ 헤더(`.inc`)에 `::mlir::LogicalResult verify();` 메서드 선언이 자동으로 생김  
PolyDialect.cpp에서 구현

```
LogicalResult EvalOp::verify() {
  return getPoint().getType().IsInteger(32)
             ? success()
             : emitOpError("argument point must be a 32-bit integer");
}
```

#### 3. Custom Trait Verifier

Verifier 규칙이 여러 연산에 공통적으로 필요하다면, 나만의 Trait를 만들어 사용  
`PolyTraits.h`에 아래와 같이 구현

```C++
template <typename ConcreteType>
class Has32BitArguments : public OpTrait::TraitBase<ConcreteType, Has32BitArguments> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    for (auto type : op->getOperandTypes()) {
      // OK to skip non-integer operand types
      if (!type.isIntOrIndex()) continue;

      if (!type.isInteger(32)) {
        return op->emitOpError()
               << "requires each numeric operand to be a 32-bit integer";
      }
    }

    return success();
  }
};
```

## [Canonicalizers and Declarative Rewrite Patterns](https://www.jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/)

`Canonicalization`은 `Folding`과 달리, 새로운 연산을 생성하여 그래프 구조를 변경  
MLIR에서 DAG-to-DAG rewriting으로 부름 (`DAG`: [link](https://llvm.org/devmtg/2024-10/slides/tutorial/MacLean-Fargnoli-ABeginnersGuide-to-SelectionDAG.pdf))  
`Canonicalization`도 TableGen + C++로 구현

> [!NOTE]  
> Canonicalization은 "최적화"라기보다는 **"깔끔하고 표준적인 형태"**로 정리하는 과정입니다.  
> 모든 컴파일러 패스가 같은 형태의 IR을 볼 수 있도록 통일성을 부여합니다.

#### 1. Canonicalizers in C++

연산에 대한 TableGen에서 `let hasCanonicalizer = 1;`을 추가하면, C++ 헤더(`.inc`)에 `static void getCanonicalizationPatterns`가 자동으로 생김  
C++로 rewrite 패턴을 구현한 뒤 `getCanonicalizationPatterns`에 추가  
아래는 합차공식의 예시

```C++
struct DifferenceOfSquares : public OpRewritePattern<SubOp> {
  DifferenceOfSquares(mlir::MLIRContext *context)
      : OpRewritePattern<SubOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    // If either arg has another use, then this rewrite is probably less
    // efficient, because it cannot delete the mul ops.
    if (!lhs.hasOneUse() || !rhs.hasOneUse()) {
      return failure();
    }

    auto rhsMul = rhs.getDefiningOp<MulOp>();
    auto lhsMul = lhs.getDefiningOp<MulOp>();
    if (!rhsMul || !lhsMul) {
      return failure();
    }

    bool rhsMulOpsAgree = rhsMul.getLhs() == rhsMul.getRhs();
    bool lhsMulOpsAgree = lhsMul.getLhs() == lhsMul.getRhs();

    if (!rhsMulOpsAgree || !lhsMulOpsAgree) {
      return failure();
    }

    auto x = lhsMul.getLhs();
    auto y = rhsMul.getLhs();

    AddOp newAdd = rewriter.create<AddOp>(op.getLoc(), x, y);
    SubOp newSub = rewriter.create<SubOp>(op.getLoc(), x, y);
    MulOp newMul = rewriter.create<MulOp>(op.getLoc(), newAdd, newSub);

    rewriter.replaceOp(op, {newMul});
    // We don't need to remove the original ops because MLIR already has
    // canonicalization patterns that remove unused ops.

    return success();
  }
};
```

#### 2. Canonicalizers in Tablegen

아래는 복소수의 (켤레 + 다항식) -> (다항식 + 켤레) 로 순서 고정하는 예시

```
include "PolyOps.td"
include "mlir/Dialect/Complex/IR/ComplexOps.td"
include "mlir/IR/PatternBase.td"

def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z)),
  (ConjOp (Poly_EvalOp $f, $z))
>;
```

아래는 합차공식의 예시

- 합차공식은 복잡해서 Pat 대신 Pattern 사용
- def HasOneUse는 f-string과 같은 용도
- hasOneUse()는 MLIR 내장 함수

```
// PolyPatterns.td
def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

// Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other uses.
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;
```
