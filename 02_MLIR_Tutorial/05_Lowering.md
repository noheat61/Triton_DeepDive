## [Dialect Conversion](https://www.jeremykun.com/2023/10/23/mlir-dialect-conversion/)

> [!IMPORTANT]  
> MLIR의 컴파일 파이프라인은 한 번에 낮아지는 게 아니라 점진적으로 축소됩니다.  
> 기존의 rewrite로 연산은 변환할 수 있지만, 타입이 꼬이기 쉽습니다.  
> `poly.add` → `arith.addi` 같은 변환에서 연산자뿐 아니라 operand/result 타입까지 일관되게 변환해야 하고, 앞뒤 IR 간 타입의 연관성에 따라 전파되기 때문에, 타입 변환은 굉장히 어려운 작업입니다.  
> MLIR에서 이런 문제를 해결하기 위해 **Dialect Conversion**을 사용합니다.

#### 1. Conversion Engine

`lib/Conversion`에 TableGen/C++로 구현

```
def PolyToStandard : Pass<"poly-to-standard"> {
  let summary = "Lower `poly` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `poly` dialect to standard MLIR, a mixture of affine,
    tensor, and arith.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tutorial::poly::PolyDialect",
    "mlir::tensor::TensorDialect",
    "mlir::scf::SCFDialect",
  ];
}
```

- 이전의 TableGen과 다르게, `let dependentDialects`가 있음
  - 다른 Pass들은 하나의 Dialect 내부에서 실행되므로, Pass가 시작될 때 이미 Dialect가 로드되어 있음
  - 변환 Pass는 하나의 Dialect 내부가 아닌 여러 Dialect를 필요로 하므로, Pass를 시작하기 전 모든 Dialect를 로드할 필요가 있음

```C++
struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    ConversionTarget target(*context);
    target.addIllegalDialect<PolyDialect>();

    RewritePatternSet patterns(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

- `ConversionTarget`: 변환의 종료 조건을 정의
  - `target.addIllegalDialect`: 이 Dialect가 완전히 사라져야 성공으로 간주
- `applyPartialConversion`: 단순 최적화와 달리, target을 만족할 때까지 변환
  - 일반 Pass에서 rewrite할 때는 `applyPatternsAndFoldGreedily` 사용했었음
  - `applyPartialConversion`에서는 illegal 연산이 남아있으면 변환 실패로 간주

#### 2. Rewrite Pattern

Pass에서 다뤘던 rewrite 패턴과 비슷하게, 변환 Pass에 대해서도 각 연산에 대해 rewrite 패턴 정의

```C++
struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    arith::AddIOp addOp = rewriter.create<arith::AddIOp>(
        op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op.getOperation(), addOp);
    return success();
  }
};
```

- `adaptor`: 변환 이후의 타입을 저장한 객체
  - `AddOp`를 TableGen으로 정의할 때, MLIR은 자동으로 **AddOpAdaptor**라는 클래스도 같이 만듦
  - `matchAndRewrite` 실행 전 이미 타입 변환된 `Value`들을 `Adaptor`에 저장
  - 타입 변환에 대해서는 고려하지 않고 rewrite 패턴을 작성할 수 있음

#### 3. Type Converter

실제 데이터의 타입 변환을 수행하는 핵심 class

- ConvertAdd뿐 아니라, poly -> arith 변환을 수행하는 대부분의 연산에 공통적으로 활용됨

```C++
class PolyToStandardTypeConverter : public TypeConverter {
 public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    // 1. 기본 규칙 (Identity Conversion)
    // "내가 모르는 타입(예: 이미 Standard인 타입)은 건드리지 말고 그대로 둬라."
    addConversion([](Type type) { return type; });

    // 2. Poly 변환 규칙
    // "!poly.poly<10> (Degree 10)" -> "tensor<10xi32> (32비트 정수 10개)"
    addConversion([ctx](PolynomialType type) -> Type {
      int degreeBound = type.getDegreeBound();
      // 32비트 정수 타입 생성 (i32)
      IntegerType elementTy = IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Signless);
      // 랭크가 있는 텐서로 변환
      return RankedTensorType::get({degreeBound}, elementTy);
    });
  }
};
```

#### 4. Conversion Engine (Fixed)

Conversion Engine에 rewrite 패턴들을 추가

```C++
struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<PolyDialect>();

    RewritePatternSet patterns(context);
    PolyToStandardTypeConverter typeConverter(context);
    patterns.add<ConvertAdd, ConvertConstant, ConvertSub, ConvertEval,
                 ConvertMul, ConvertFromTensor, ConvertToTensor>(typeConverter,
                                                                 context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

- `patterns.add`: `ConvertAdd` 클래스의 생성자에 `typeConverter`가 자동으로 전달
  - 패턴 내부에서 `matchAndRewrite`가 실행되기 전, `typeConverter`를 사용하여 입력값(Operands)을 미리 변환
  - `adaptor.getLhs()`를 통해 이미 변환된 텐서 값을 편하게 꺼내 쓸 수 있음

#### 5. 예외처리

함수 입출력(Signature), 리턴, 호출, 분기(Branch) 또한 변환해주어야 함  
관련 함수가 있으므로, 아래처럼 변환 패턴을 추가

```C++
struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;

  void runOnOperation() override {
    ...

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    ...
  }
};

```

#### 6. 테스트

```shell
bazel run //tools:tutorial-opt -- --poly-to-standard $(pwd)/tests/poly_to_standard.mlir
```

#### 7. Materialization hooks (option)

튜토리얼에서는 잘 수행되었지만, 변환 과정에서 문제가 있다면 원래의 Dialect(poly)로 rollback 후 연산을 수행해야 함  
typeConverter에 연산 전후 사용할 수 있는 Materialization hooks 추가

```C++
class PolyToStandardTypeConverter : public TypeConverter {
 public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    addConversion([ctx](PolynomialType type) -> Type {
      int degreeBound = type.getDegreeBound();
      IntegerType elementTy = IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Signless);
      return RankedTensorType::get({degreeBound}, elementTy);
    });

    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<poly::FromTensorOp>(loc, type, inputs[0]);
    });

    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<poly::ToTensorOp>(loc, type, inputs[0]);
    });
  }
};
```

문제가 있을 경우 `from_tensor`, `to_tensor`를 호출하도록 설정  
아래와 같이 IR 수정됨

```
  func.func @test_lower_many(%arg0: tensor<10xi32>, %arg1: i32) -> i32 {
    %2 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %cst_0) -> (tensor<10xi32>) {
      %7 = scf.for %arg4 = %c0 to %c10 step %c1 iter_args(%arg5 = %arg3) -> (tensor<10xi32>) {
         ...
      }
      scf.yield %7 : tensor<10xi32>
    }
    %3 = poly.from_tensor %2 : tensor<10xi32> -> !poly.poly<10>
    %4 = poly.sub %3, %0 : !poly.poly<10>
    %5 = poly.to_tensor %4 : !poly.poly<10> -> tensor<10xi32>
    %c1_1 = arith.constant 1 : index
    ...
  }
```

## [Lowering through LLVM](https://www.jeremykun.com/2023/11/01/mlir-lowering-through-llvm/)

MLIR의 IR은 최종적으로 LLVM Dialect로 Lowering해야 함  
그 다음 LLVM Dialect -> LLVM IR은 codegen에서 수행  
튜토리얼에서는 위에서 변환된 arith -> LLVM으로 변환하는 과정을 보여줌

> [!NOTE]  
> 튜토리얼에서는 아래 순서를 찾기 위한 시행착오의 과정을 보여줍니다.  
> 여기서는 최종적인 순서를 보여주는 것이기 때문에 어떻게 Pass의 순서가 구성되었는지 궁금하다면 튜토리얼을 읽는 것을 추천합니다.

> [!NOTE]  
> 이 단계는 CPU를 위한 단계이므로, 모든 MLIR 최적화에서 사용되지 않습니다. Triton의 경우는 TTIR -> TTGIR -> LLVM으로 자신들의 Dialect에서 바로 LLVM 변환을 수행하였습니다.  
> 따라서 아래 순서는 학습을 위한 참고사항으로 생각해주세요.

#### 1. Tensor to Linalg

연산(`tensor.splat`, `arith.add` 등)을 선형 대수 연산(linalg)으로 변환

- `elementwise-to-linalg`: arith 연산 등을 linalg.generic으로 변환
- `tensor-to-linalg`: tensor 관련 연산을 linalg로 변환

#### 2. Bufferization

**값 중심**에서 **메모리 중심**으로 넘어가는, **가장 중요한 단계**

- `one-shot-bufferization`: 텐서를 메모리 버퍼(memref)로 할당하고 변환

#### 3. Linalg to Loops to CF

linalg는 "행렬 곱셈해라" 같은 고수준 명령  
컴퓨터가 이해하는 for 루프와 if/else 분기로 변환

- `linalg-to-loops`: linalg 연산을 scf.for (Structured Control Flow)로 변환
- `scf-to-cf`: scf.for를 cf.br (Unstructured Control Flow - goto문 같은 것)로 변환

#### 4. Dialect to LLVM

- `arith-to-llvm`: 산술 연산 변환 (i32 -> llvm.i32)
- `cf-to-llvm`: 분기문 변환
- `func-to-llvm`: 함수 껍데기와 인자/리턴 타입을 변환
