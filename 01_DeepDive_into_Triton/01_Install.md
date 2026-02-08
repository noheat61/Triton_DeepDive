## Triton 설치

- 기본적으로 `pip install triton`을 통해 사전 빌드된 바이너리를 설치할 수 있음
- 그러나 Triton 컴파일러의 동작 원리를 MLIR과 연결하여 이해하기 위해, 호환되는 버전의 LLVM을 직접 빌드하고 이를 연동하여 Triton을 설치
- `llvm-project`의 PATH는 사용자가 원하는 곳으로 수정할 수 있음

```
git clone https://github.com/triton-lang/triton.git
cd triton

git clone https://github.com/llvm/llvm-project
cd llvm-project
git checkout $(cat ../cmake/llvm-hash.txt)

mkdir build
cd build
cmake  ../llvm -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"

# 오래 걸림
# 메모리 부족 시 -j1으로 조정 권장
ninja -j2

cd ../..
export LLVM_BUILD_DIR=$PWD/llvm-project/build

# 오래 걸림
# conda 등을 활용하여 가상환경 세팅 추천
MAX_JOBS=2 LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib LLVM_SYSPATH=$LLVM_BUILD_DIR pip install -v -e .
```
