## [Defining a New Dialect](https://www.jeremykun.com/2023/08/21/mlir-defining-a-new-dialect/)

32비트 Unsigned Integer 계수를 가지는 다항식 연산을 위한 `poly` Dialect를 정의

#### 1. Dialect 정의 (`PolyDialect.td`)

Pass 정의와 유사하게, TableGen을 활용하여 Dialect의 메타데이터를 작성

```
def Poly_Dialect : Dialect {
  let name = "poly";
  let summary = "A dialect for polynomial math";
  let description = [{
    The poly dialect defines types and operations for single-variable
    polynomials over integers.
  }];

  let cppNamespace = "::mlir::tutorial::poly";

  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}
```

Pass와 유사하게, tutorial-opt.cpp에서 Dialect 등록  
아래 명령어를 실행해서, Dialect 목록에 `poly`가 있는지 확인

```
bazel run //tools:tutorial-opt -- --help | head -n 3
```

#### 2. Type 정의 (`PolyTypes.td`)

Dialect 안에 들어갈 자료형(Type)을 정의  
Type에 대해서도 TableGen을 활용하여 작성

```
class Poly_Type<string name, string typeMnemonic> : TypeDef<Poly_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Polynomial : Poly_Type<"Polynomial", "poly"> {
  ...
}
```

구조:`Poly_Dialect(def)` → `Poly_Type(class)` → `Polynomial(def)`

- Poly_Type(class): poly dialect 내의 타입을 나타내는 클래스
- Polynomial(def): 실제 타입

```
def Polynomial : Poly_Type<"Polynomial", "poly"> {
  let summary = "A polynomial with u32 coefficients";

  let description = [{
    A type for polynomials with integer coefficients in a single-variable polynomial ring.
  }];

  let parameters = (ins "int":$degreeBound);
  let assemblyFormat = "`<` $degreeBound `>`";
}
```

- `def Polynomial`: TableGen 내부 변수명 (C++ 코드 생성 후에는 쓰이지 않음)
- `Polynomial`: 생성될 C++ 클래스 이름 (PolynomialType이 됨)
- `poly`: Mnemonic(약어). MLIR 코드에서 !poly.poly처럼 쓰일 때 뒤에 오는 이름

#### 3. 파일 생성 및 빌드

Pass는 .h.inc만 생성하지만, Dialect와 Type은 **.cpp.inc**도 생성되므로 조금 더 복잡함

- Pass는 `Passes.td` -> `Passes.h.inc` 생성 -> `Passes.h`에서 가져옴
- Type은 `PolyTypes.td` -> `PolyTypes.h.inc` / `PolyTypes.cpp.inc` 생성
  - BUILD에서 target으로 `PolyTypes.cpp.inc`도 추가해야 정상적으로 생성됨

```
namespace mlir {
namespace tutorial {
namespace poly {

void PolyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Poly/PolyTypes.cpp.inc"
      >();
}
```

- `PolyDialect.cpp`에서 `Polytypes.cpp.inc`/`Polyops.cpp.inc` 등등 가져와서 사용
- `#define GET_TYPEDEF_LIST`: `PolyTypes.cpp.inc` 파일은 방대하므로, 그 중에서 `typedef` 목록만 뽑아오기 위해 사용하는 스위치

#### 4. poly 타입 매개변수 추가

단순히 "다항식이다"라고 정의하는 것을 넘어, **최대 차수**를 타입의 속성으로 포함

- 단일 변수 다항식만을 고려하므로, 핵심 정보는 **계수**와 **차수**뿐
- 다항식의 차수는 가변적이지만, 컴파일 타임에 메모리 구조를 확정하기 위해 **최대 차수**를 설정

TableGen을 활용, 단 두줄만 추가하면 기능들이 자동 생성됨

```
let parameters = (ins "int":$degreeBound);
let assemblyFormat = "`<` $degreeBound `>`";
```

- Getter 함수 생성: 타입 객체에서 `type.getDegreeBound()`를 호출하여 설정된 차수 값을 꺼내쓸 수 있음
- Factory 함수 업데이트: `PolynomialType::get(context)` -> `PolynomialType::get(context, int degreeBound)`
- Parser/Printer 지원: 컴파일러가 !poly.poly<10> 같은 문법을 자동으로 이해하고 출력

#### 5. Operation 추가 (`PolyOps.td`)

타입이 정의되었으니, 이제 그 타입을 사용하는 **연산**을 정의  
dialect랑 type 모두 include해야 함

```
include "PolyDialect.td"
include "PolyTypes.td"

def Poly_AddOp : Op<Poly_Dialect, "add"> {
  let summary = "Addition operation between polynomials.";
  let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}
```

## [Using Traits](https://www.jeremykun.com/2023/09/07/mlir-using-traits/)

> [!IMPORTANT]
> MLIR에서 최적화를 수행하려면, 컴파일러가 IR이 어떤 **성질**을 가지는지 판단할 수 있어야 합니다. 하지만 실제 IR은 하나의 dialect가 아니라 여러 dialect가 섞인 조합으로 구성되어 있습니다.  
> 특정 dialect에 종속되지 않고 IR 단위로 일반적인 최적화를 적용하려면, 각 op나 type이 어떤 조건을 만족하는지 매번 직접 검사해야 합니다.  
> 이 문제를 해결하기 위해 MLIR에서 사용하는 것이 **Trait**입니다.
>
> - Dialect 제작자가 나는 이런 성격을 가졌어라고 **스티커**를 붙이는 것과 같음
> - 스티커가 붙어 있으면 MLIR 컴파일러가 "아, 넌 안전한 녀석이구나?" 하고 최적화를 수행
> - 최적화에 필요한 조건의 판단 결과를 op에 미리 전달해 두는 공식적인 인터페이스

Loop Invariant Code Motion (LICM): 루프 안에서 변하지 않는 값을 밖으로 빼는 MLIR 최적화  
연산에 **Pure**라는 Trait을 추가하면 안심하고 LICM 수행

```
 include "PolyDialect.td"
 include "PolyTypes.td"
+include "mlir/Interfaces/SideEffectInterfaces.td"


-class Poly_BinOp<string mnemonic> : Op<Poly_Dialect, mnemonic> {
+class Poly_BinOp<string mnemonic> : Op<Poly_Dialect, mnemonic, [Pure]> {
   let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
```

`Pure`는 두 가지 Trait의 조합

1. `NoMemoryEffect`: "나는 메모리에 손대지 않아요(Read/Write 없음)."
2. `AlwaysSpeculatable`: "나를 미리 실행해도 컴퓨터가 터지거나 결과가 달라지지 않아요."

> [!WARNING]  
> 실제로 부작용이 있는 연산인데 거짓말로 `Pure`를 붙이면, 최적화 과정에서 연산 순서가 뒤섞여 프로그램이 망가질 수 있습니다.  
> Trait를 붙여도 괜찮은지에 대한 판단은 Dialect 개발자가 꼼꼼하게 수행해야 합니다.

Pure 외에 자주 사용되는 Trait들

- `ElementwiseMappable`: 배열(Vector/Tensor) 단위의 연산 최적화
  - `Polynomial`을 단일 값과 컨테이너 모두 수용 가능한 `PolyOrContainer`로 수정해야 함
- `SameOperandsAndResultType`: 입력값들과 결과값의 타입이 모두 같아야 함을 보장

> [!NOTE]  
> MLIR Trait은 친절한 공식 문서가 거의 없습니다.  
> 원하는 최적화를 적용하고 싶을 때, 최적화 패스의 소스 코드를 열어 어떤 Trait을 검사하는지 직접 확인하고 Dialect에 적용해야 합니다.
