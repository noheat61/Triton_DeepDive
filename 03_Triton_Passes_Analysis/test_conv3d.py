import torch
import triton
import triton.language as tl
import triton.testing

@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, C_in, D, H, W,
    C_out, KD, KH, KW,
    D_out, H_out, W_out,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_INT8: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    if IS_INT8:
        ACC_TYPE = tl.int32
    else:
        ACC_TYPE = tl.float32
        
    pid = tl.program_id(0)
    
    stride_in_w = C_in
    stride_in_h = W * C_in
    stride_in_d = H * W * C_in
    stride_in_n = D * H * W * C_in
    
    stride_w_cout = 1
    stride_w_cin = C_out
    stride_w_kw = C_in * C_out
    stride_w_kh = KW * C_in * C_out
    stride_w_kd = KH * KW * C_in * C_out

    num_pid_n = tl.cdiv(C_out, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    spatial_size = D_out * H_out * W_out
    n_batch = offs_m // spatial_size
    remainder = offs_m % spatial_size
    
    d_out = remainder // (H_out * W_out)
    remainder = remainder % (H_out * W_out)
    h_out = remainder // W_out
    w_out = remainder % W_out

    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    mask_m = offs_m < (N * spatial_size)
    mask_n = offs_n < C_out

    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                
                in_d = d_out * stride_d + kd - padding_d
                in_h = h_out * stride_h + kh - padding_h
                in_w = w_out * stride_w + kw - padding_w
                
                is_valid = (in_d >= 0) & (in_d < D) & \
                           (in_h >= 0) & (in_h < H) & \
                           (in_w >= 0) & (in_w < W)
                
                input_base = input_ptr + (
                    n_batch * stride_in_n + 
                    in_d * stride_in_d + 
                    in_h * stride_in_h + 
                    in_w * stride_in_w
                )
                
                weight_base = weight_ptr + (
                    kd * stride_w_kd + 
                    kh * stride_w_kh + 
                    kw * stride_w_kw
                )

                for k in range(0, C_in, BLOCK_K):
                    offs_k_curr = k + offs_k
                    mask_k = offs_k_curr < C_in
                    
                    ptr_input = input_base[:, None] + offs_k_curr[None, :]
                    mask_in = mask_m[:, None] & is_valid[:, None] & mask_k[None, :]
                    a = tl.load(ptr_input, mask=mask_in, other=0.0)
                    
                    ptr_weight = weight_base + \
                                 offs_k_curr[:, None] * stride_w_cin + \
                                 offs_n[None, :] * stride_w_cout
                    mask_w = mask_k[:, None] & mask_n[None, :]
                    b = tl.load(ptr_weight, mask=mask_w, other=0.0)
                    
                    if IS_INT8:
                        accum += tl.dot(a, b, out_dtype=tl.int32)
                    else:
                        accum += tl.dot(a, b, allow_tf32=True)

    offs_out = offs_m[:, None] * C_out + offs_n[None, :]
    
    if IS_INT8:
        c = accum
    elif IS_FP16:
        c = accum.to(tl.float16)
    else:
        c = accum.to(tl.float32)

    tl.store(output_ptr + offs_out, c, mask=mask_m[:, None] & mask_n[None, :])

# def run_triton_conv3d(x, w, padding, dtype):
#     N, D, H, W, C_in = x.shape
#     C_out, _, KD, KH, KW = w.shape
    
#     w_tr = w.permute(2, 3, 4, 1, 0).contiguous()
    
#     padding_d, padding_h, padding_w = padding
#     D_out = D + 2 * padding_d - KD + 1
#     H_out = H + 2 * padding_h - KH + 1
#     W_out = W + 2 * padding_w - KW + 1
    
#     out_dtype = dtype
#     if dtype == torch.int8:
#         out_dtype = torch.int32

#     y = torch.empty((N * D_out * H_out * W_out, C_out), device=x.device, dtype=out_dtype)
    
#     BLOCK_M = 128
#     BLOCK_N = 64
#     BLOCK_K = 32
    
#     is_int8 = (dtype == torch.int8)
#     is_fp16 = (dtype == torch.float16)
    
#     grid = (triton.cdiv(N * D_out * H_out * W_out, BLOCK_M) * triton.cdiv(C_out, BLOCK_N),)
    
#     conv3d_kernel[grid](
#         x, w_tr, y,
#         N, C_in, D, H, W,
#         C_out, KD, KH, KW,
#         D_out, H_out, W_out,
#         1, 1, 1,
#         padding_d, padding_h, padding_w,
#         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
#         IS_INT8=is_int8, IS_FP16=is_fp16,
#         num_warps=4, num_stages=2
#     )
    
#     return y.view(N, D_out, H_out, W_out, C_out)

# def run_benchmark():
#     N, C_in, D, H, W = 1, 64, 16, 256, 256
#     C_out, KD, KH, KW = 64, 3, 3, 3
#     padding = (1, 1, 1)
#     device = "cuda"
    
#     print(f"--- Configuration ---")
#     print(f"Input: ({N}, {C_in}, {D}, {H}, {W})")
#     print(f"Kernel: ({C_out}, {C_in}, {KD}, {KH}, {KW})")
#     print("-" * 75)
#     print(f"{'Mode':<10} | {'Dtype':<6} | {'Correct?':<8} | {'Max Diff':<10} | {'Time (ms)':<10} | {'Speedup':<8}")
#     print("-" * 75)

#     torch.manual_seed(0)
#     x_fp32 = torch.randn((N, C_in, D, H, W), device=device, dtype=torch.float32)
#     w_fp32 = torch.randn((C_out, C_in, KD, KH, KW), device=device, dtype=torch.float32)
    
#     torch.backends.cudnn.allow_tf32 = True
#     y_ref_fp32 = torch.nn.functional.conv3d(x_fp32, w_fp32, padding=padding)
    
#     x_tr_fp32 = x_fp32.permute(0, 2, 3, 4, 1).contiguous()
#     y_tr_fp32 = run_triton_conv3d(x_tr_fp32, w_fp32, padding, torch.float32).permute(0, 4, 1, 2, 3)
    
#     diff_fp32 = (y_ref_fp32 - y_tr_fp32).abs().max().item()
#     is_close_fp32 = torch.allclose(y_ref_fp32, y_tr_fp32, atol=1e-3)
    
#     ms_torch_fp32 = triton.testing.do_bench(lambda: torch.nn.functional.conv3d(x_fp32, w_fp32, padding=padding))
#     ms_triton_fp32 = triton.testing.do_bench(lambda: run_triton_conv3d(x_tr_fp32, w_fp32, padding, torch.float32))
    
#     print(f"{'PyTorch':<10} | {'FP32':<6} | {'Ref':<8} | {'-':<10} | {ms_torch_fp32:.4f}     | {'1.00x':<8}")
#     print(f"{'Triton':<10} | {'FP32':<6} | {str(is_close_fp32):<8} | {diff_fp32:.6f}   | {ms_triton_fp32:.4f}     | {ms_torch_fp32/ms_triton_fp32:.2f}x")
#     print("-" * 75)

#     x_fp16 = x_fp32.half()
#     w_fp16 = w_fp32.half()
    
#     y_ref_fp16 = torch.nn.functional.conv3d(x_fp16, w_fp16, padding=padding)
    
#     x_tr_fp16 = x_fp16.permute(0, 2, 3, 4, 1).contiguous()
#     y_tr_fp16 = run_triton_conv3d(x_tr_fp16, w_fp16, padding, torch.float16).permute(0, 4, 1, 2, 3)
    
#     diff_fp16 = (y_ref_fp16 - y_tr_fp16).abs().max().item()
#     is_close_fp16 = torch.allclose(y_ref_fp16, y_tr_fp16, atol=1e-1, rtol=1e-2) 
    
#     ms_torch_fp16 = triton.testing.do_bench(lambda: torch.nn.functional.conv3d(x_fp16, w_fp16, padding=padding))
#     ms_triton_fp16 = triton.testing.do_bench(lambda: run_triton_conv3d(x_tr_fp16, w_fp16, padding, torch.float16))
    
#     print(f"{'PyTorch':<10} | {'FP16':<6} | {'Ref':<8} | {'-':<10} | {ms_torch_fp16:.4f}     | {'1.00x':<8}")
#     print(f"{'Triton':<10} | {'FP16':<6} | {str(is_close_fp16):<8} | {diff_fp16:.6f}   | {ms_triton_fp16:.4f}     | {ms_torch_fp16/ms_triton_fp16:.2f}x")
#     print("-" * 75)

#     x_int8 = torch.randint(-3, 3, (N, C_in, D, H, W), device=device, dtype=torch.int8)
#     w_int8 = torch.randint(-3, 3, (C_out, C_in, KD, KH, KW), device=device, dtype=torch.int8)
    
#     x_i8_f = x_int8.float()
#     w_i8_f = w_int8.float()
#     y_ref_int8_float = torch.nn.functional.conv3d(x_i8_f, w_i8_f, padding=padding)
    
#     x_tr_int8 = x_int8.permute(0, 2, 3, 4, 1).contiguous()
#     y_tr_int8 = run_triton_conv3d(x_tr_int8, w_int8, padding, torch.int8).permute(0, 4, 1, 2, 3)
    
#     y_tr_int8_float = y_tr_int8.float()
    
#     diff_int8 = (y_ref_int8_float - y_tr_int8_float).abs().max().item()
#     is_close_int8 = diff_int8 == 0
    
#     ms_triton_int8 = triton.testing.do_bench(lambda: run_triton_conv3d(x_tr_int8, w_int8, padding, torch.int8))
    
#     print(f"{'PyTorch':<10} | {'INT8':<6} | {'N/A*':<8} | {'N/A':<10} | {'N/A':<10} | {'-':<8}")
#     print(f"{'Triton':<10} | {'INT8':<6} | {str(is_close_int8):<8} | {diff_int8:.1f}        | {ms_triton_int8:.4f}     | vs FP16: {ms_torch_fp16/ms_triton_int8:.2f}x")

# if __name__ == "__main__":
#     run_benchmark()