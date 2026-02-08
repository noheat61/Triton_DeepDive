## Triton Frontend Analysis

- compile.py를 실행했을 때 거치는 frontend 과정을 단계별로 분석

### python/triton/tools/compile.py

#### importlib.util

```python
spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
```

- 컴파일할 대상 프로그램을 가져오는 단계
- `exec_module(mod)`: 모듈 파일을 쭉 한 번 읽으면서 실행
  - 실제 커널 컴파일은 발생하지 않음
  - 단지 Python 레벨에서 커널 정의와 메타정보를 로딩하는 단계

#### Kernel metadata configuration

```python
hints = {(i, ): constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
hints = {k: v for k, v in hints.items() if v is not None}
constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
constants = {k: v for k, v in constants.items() if v is not None}
for key, value in hints.items():
   if value == 1:
      constants[kernel.arg_names[key[0]]] = value
signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature)}
for key in constants:
   signature[key] = 'constexpr'
const_sig = 'x'.join([str(v) for v in constants.values()])
doc_string = [f"{k}={v}" for k, v in constants.items()]
doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]
# compile ast into cubin
for h in hints.values():
   assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
attrs = {k: [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}
kernel.create_binder()
src = kernel.ASTSource(fn=kernel, constexprs=constants, signature=signature, attrs=attrs)
```

- signiture 분석해서 hint 추출 / 상수 변환
  - ex) signature = ["A", "B:16", "C:1", "D", "64"]
  - => hints = {(1,): 16, (2,): 1}
  - => constants = {'BLOCK_SIZE': 64}

> [!QUESTION]  
> hints로는 1 or 16만 가능한데, 두 개의 값이 가지는 의미를 찾아봐도 잘 모르겠습니다...  
> 그리고 1은 왜 constants도 되고 hints도 되는지...  
> 나중에 알게 되면 이 부분 수정

#### get target

```python
target = triton.backends.compiler.GPUTarget(*args.target.split(":")) \
   if args.target else triton.runtime.driver.active.get_current_target()
backend = triton.compiler.make_backend(target)
```

- 내부 동작 흐름을 분석하면
  - `third_party/nvidia/backend/driver.py`와 `python/triton/backends/driver.py`의 GPUDriver에서 실행
  - target에 대한 정보는 모두 torch를 활용해 가져옴

### python/triton/compiler/compiler.py

#### 캐시 확인

```python
extra_options = src.parse_options()
options = backend.parse_options(dict(options or dict(), **extra_options))
# create cache manager
env_vars = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
key = get_cache_key(src, backend, options, env_vars=env_vars)
if knobs.runtime.add_stages_inspection_hook is not None:
   inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
   key += inspect_stages_key
hash = hashlib.sha256(key.encode("utf-8")).hexdigest()

...

if not always_compile and metadata_path is not None:
   # cache hit!
   res = CompiledKernel(src, metadata_group, hash)
   if compilation_listener:
      compilation_listener(
            src=src,
            metadata=res.metadata._asdict(),
            metadata_group=metadata_group,
            times=timer.end(),
            cache_hit=True,
      )
   return res
```

- 소스 코드(`src`), 백엔드 설정(`backend`), 컴파일 옵션(`options`), 그리고 환경 변수(`env_vars`)까지 모두 포함하여 SHA256 해시 생성
- cache manager에 있을 경우(cache hit), 컴파일 파이프라인을 스킵하고 `CompiledKernel`만 복원

#### 메타데이터 초기화

```python
# initialize metadata
metadata = {
   "hash": hash,
   "target": target,
   **options.__dict__,
   **env_vars,
}
metadata["triton_version"] = __version__
```

- "이 커널이 어떤 조건에서 만들어졌는지"를 기록

#### 컴파일 스테이지 구성

```python
# run compilation pipeline  and populate metadata
stages = dict()
backend.add_stages(stages, options, src.language)
first_stage = list(stages.keys()).index(src.ext)
# when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
if ir_source:
   first_stage += 1

# For IRSource, we have already grabbed the context + called both
# ir.load_dialects and backend.load_dialects.
if not isinstance(src, IRSource):
   context = ir.context()
   ir.load_dialects(context)
   backend.load_dialects(context)
```

- `third_party/nvidia/backend/compiler.py`에서 `add_stage` 호출
  - `make_ttir`, `make_ttgir`, `make_llir` 등 모든 레벨에서의 Lowering Pass를 정의

- `load_dialects`는 MLIR의 코드 호출
  - `ir.load_dialects(context)` -> `python/src/ir.cc`
  - `backend.load_dialects(context)` -> `third_party/nvidia/triton_nvidia.cc`

#### TTIR 생성

```python
codegen_fns = backend.get_codegen_implementation(options)
module_map = backend.get_module_map()
try:
   module = src.make_ir(target, options, codegen_fns, module_map, context)
except Exception as e:
   filter_traceback(e)
   raise

def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
   from .code_generator import ast_to_ttir
   return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns, module_map=module_map)
```

- Python AST 기반 Triton kernel -> TTIR로 변환
- 자세한 과정은 밑에서 분석

#### 이후 단계

- 이 다음 과정부터는 Backend 최적화

### python/triton/compiler/code_generator.py

```python
def ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
    arg_types = [None] * len(fn.arg_names)

    for k, v in src.signature.items():
        idx = fn.arg_names.index(k)
        arg_types[idx] = str_to_ty(v, None)

    def apply_constexpr_types(argument, indices, value):
        index = indices.pop()
        if len(indices) == 0:
            if isinstance(argument, list):
                argument[index] = constexpr(value).type
            else:
                argument.types[index] = constexpr(value).type
        else:
            apply_constexpr_types(argument[index], indices, value)

    for path, value in src.constants.items():
        apply_constexpr_types(arg_types, list(path)[::-1], value)

    prototype = ASTFunction([], arg_types, src.attrs)
    file_name, begin_line = get_jit_fn_file_line(fn)
    # query function representation
    from collections import namedtuple
    leaves = filter(lambda v: len(v) == 1, src.constants)
    constants = {fn.arg_names[i[0]]: src.constants[i] for i in leaves}
    signature = src.signature
    proxy = namedtuple("SpecializationProxy", ["constants", "signature"])(constants, signature)
    generator = CodeGenerator(context, prototype, gscope=fn.get_capture_scope(), function_name=fn.repr(proxy),
                              jit_fn=fn, is_kernel=True, file_name=file_name, begin_line=begin_line, options=options,
                              codegen_fns=codegen_fns, module_map=module_map, module=module, is_gluon=fn.is_gluon())
    generator.visit(fn.parse())
    module = generator.module
    # module takes ownership of the context
    module.context = context
    if not module.verify():
        if not fn.is_gluon():
            print(module)
        raise RuntimeError("error encountered during parsing")
    return module
```

1. signature 파싱 -> 자료형 지정

- `*fp32`와 같이 문자열로 매개변수가 전달되었기 때문에, 이를 triton 변수로 매핑

2. 상수 고정 + 함수 정보 제공

3. `CodeGenerator` 생성

- `ast.NodeVisitor` 상속, `ast`는 파이썬 내부 라이브러리
- CodeGenerator는 파이썬 코드를 트리(AST)로 순회

4. Python AST 순회 → TTIR 생성

- 위에서 아래로 AST 순회
- 타입별로 `visit_*` 메서드 호출
  - `x` -> `visit_Name`
  - `a + b` -> `visit_BinOp`
  - `c = ...` -> `visit_Assign`
  - `if cond:` -> `visit_If`

> [!NOTE]  
> 본 문서는 전체적인 컴파일 흐름을 파악하는 데 중점을 두었습니다.  
> Frontend 단계의 구체적인 알고리즘은 `generator.visit()` 내부를 참고하시기 바랍니다.
