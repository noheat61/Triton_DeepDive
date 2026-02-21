## 예시 입력

### 1. `add_kernel`

```shell
python3 python/triton/tools/compile.py --kernel-name add_kernel --signature "*fp32,*fp32,*fp32,i32,64" --grid=1024,1024,1024 test_add_kernel.py

# Signature:
# - *fp32: x_ptr (입력1)
# - *fp32: y_ptr (입력2)
# - *fp32: output_ptr (출력)
# - i32: n_elements (요소 개수)
# - 64: BLOCK_SIZE (constexpr)
```

### 2. Conv3D (test_conv3d.py)

> [!NOTE]  
> Pytorch와 동일한 결과임을 보이기 위해, test_conv3d.py에 비교하는 코드를 작성하였습니다.  
> 주석을 해제하고 코드를 실행할 때, FP32에서는 서로 일치하지 않다고 할 수 있습니다.  
> 이는 triton과 Pytorch의 tf32 연산 순서 차이로 인한 오차 누적으로 보이며, 둘 모두 tf32=False로 수정하면 일치하는 결과를 얻을 수 있습니다.

```shell
python3 python/triton/tools/compile.py \
  --kernel-name conv3d_kernel \
  --signature "*i8,*i8,*i32,\
  i32,i32,i32,i32,i32,\
  i32,i32,i32,i32,\
  i32,i32,i32,\
  i32,i32,i32,\
  i32,i32,i32,\
  128,64,32,\
  1,0" \
  --grid=512,1,1 \
  test_conv3d.py

# Signature:
# - *i8: input_ptr
# - *i8: weight_ptr
# - *T: output_ptr (FP32/FP16/INT8, dtype는 IS_INT8/IS_FP16 경로에 따라 결정)
# - i32: N, C_in, D, H, W
# - i32: C_out, KD, KH, KW
# - i32: D_out, H_out, W_out
# - i32: stride_d, stride_h, stride_w
# - i32: padding_d, padding_h, padding_w
# - 128: BLOCK_M (constexpr)
# - 64: BLOCK_N (constexpr)
# - 32: BLOCK_K (constexpr)
# - 1: IS_INT8 (constexpr, 1=true)
# - 0: IS_FP16 (constexpr, 1=true)
```

### 3. [msda-triton](https://github.com/rziga/msda-triton) (Forward)

```shell
python3 python/triton/tools/compile.py \
  --kernel-name triton_multi_scale_deformable_attention_fwd_kernel \
  --signature "*fp32,*fp32,*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,64,4,4,0,0" \
  --grid=100,2,8 \
  test_msda.py

# Signature:
# - *fp32: out_ptr (출력)
# - *fp32: img_ptr (이미지 pyramid)
# - *fp32: sampling_points_ptr (샘플링 포인트)
# - *fp32: attention_weights_ptr (attention weights)
# - *fp32: shapes_ptr (pyramid level shapes)
# - i32: B=2 (batch)
# - i32: I=10000 (image pyramid pixels)
# - i32: C=32 (channels)
# - i32: N=100 (queries)
# - i32: H=8 (heads)
# - i32: L=4 (pyramid levels)
# - i32: P=4 (points per level)
# - 64: BLOCK_SIZE_C (constexpr)
# - 4: BLOCK_SIZE_L (constexpr)
# - 4: BLOCK_SIZE_P (constexpr)
# - 0: PADDING_MODE (constexpr, 0=zeros, 1=border)
# - 0: ALIGN_CORNERS (constexpr, 0=false, 1=true)
# Grid: [N=100, B=2, H=8]
```

## 분석 방법

third_party/nvidia/backend/compiler.py의 make_ttir(236번째 줄)에

```python
import ipdb; ipdb.set_trace()
```

추가해서 breakpoint 만들고, pm.run 전후의 mod.dump()을 보고 IR의 변화 관찰
