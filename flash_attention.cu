#include "torch/types.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdlib.h>
#include <float.h>

void attention_forward_cpu(float* out, const float* inp, int B, int T, int C, int NH) {
    // inp: (B, T, 3C)
    // attention : (B, NH, T, T)
    // pre_att: (B, NH, T, T)
    // out: (B, T, C)
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0 / sqrtf(hs);

    float* pre_att = (float*) malloc(B * T * T * NH * sizeof(float));
    float* attention = (float*) malloc(B * T * T * NH * sizeof(float));

    for (int b=0; b < B; ++b) {
        for (int t=0; t < T; ++t) {
            for (int h=0; h < NH; ++h) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = pre_att + b * T * T * NH + t * T + h * T * T;
                float* attention_bth = attention + b * T * T * NH + t * T + h * T * T;

                // 1. calculate query dot key and max value
                float max_val = -10000.0f;
                for (int t2=0; t2 <= t; ++t2) {
                    const float* key_t = inp + b * T * C3 + t * C3 + h * hs + C;

                    float sum = 0.0f;
                    for (int i=0; i < hs; ++i) {
                        sum += query_t[i] * key_t[i];
                    }

                    sum *= scale;

                    max_val = std::max(max_val, sum);
                    preatt_bth[t2] = sum;
                }
                // 2. calculate exp sum.
                float exp_sum = 0.0f;
                for (int t2=0; t2 <= t; ++t2) {
                    float expv = expf(preatt_bth[t2] - max_val);
                    exp_sum += expv;
                    attention_bth[t2] = expv;
                }
                float inv_sum = exp_sum == 0.0f? 0.0f : 1.0f / exp_sum;

                // 3. calculate softmax
                for (int t2=0; t2 <= t; ++t2) {
                    attention_bth[t2] *= inv_sum;
                }
                // 4. calculate output P dot V
                float* out_t = out + b * T * C + t * C + h * hs;
                for (int i=0; i < hs; ++i) out_t[i] = 0.0f;
                for (int t2=0; t2 <= t; ++t2) {
                    const float* value_t = inp + b * T * C3 + t * C3 + h * hs + 2 * C;
                    float att = attention_bth[t2];
                    for (int i=0; i < hs; ++i) {
                        out_t[i] += att * value_t[i];
                    }
                }
            }
        }
    }
}

__global__ void attention_query_dot_key_kernel(
    float* pre_att, 
    const float* inp,
    int B,
    int T,
    int C,
    int NH
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T * T;
    if (idx >= total_threads) return;
    
    int t2 = idx % T;
    int t = (idx / T) % T;
    int h = (idx / T / T) % NH;
    int b = (idx / T / T / NH) % B;

    int head_size = C / NH;
    int C3 = C * 3;
    const float* query_t = inp + b * T * C3 + t * C3 + h * head_size;
    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * head_size + C;

    float val = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        val += query_t[i] * key_t2[i];
    }

    val /= sqrtf(head_size);
    pre_att[idx] = val;
}

__global__ void attention_softmax_kernel(float* attention, const float* inp, int B, int T, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;
    if (idx >= total_threads) return;

    int h = (idx / T / T) % NH;
    int t = idx % T;
    int b = (idx / T / NH) % B;

    float* attention_bth = attention + b * T * T * NH + h * T * T + t * T;
    const float* preatt_bth = inp + b * T * T * NH + h * T * T + t * T;

    float max_val = -1e20;
    for (int i=0; i < t; ++i) {
        max_val = max(max_val, preatt_bth[i]);
    }
    float sum = 0.0f;
    for (int i=0; i <= t; ++i) {
        float expv = expf(preatt_bth[i] - max_val);
        sum += expv;
        attention_bth[i] = expv;
    }
    float expsum_idv = sum == 0.0f ? 0.0f :  1 / sum;

    for (int i=0; i < T; ++i) {
        attention_bth[i] *= expsum_idv;
    }
}


__global__ void attention_output_kernel(float* out, const float* attention, const float* inp, int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T;
    if (idx >= total_threads) return;
    int b = (idx / T / NH) % B;
    int t = (idx / NH) % T;
    int h = idx % NH;

    int C3 = 3 * C;
    int hs = C / NH;

    float* out_split = out + b * T * C + t * C + h * hs;
    float* attention_bth = attention + b * T * T * NH + h * T * T + t * T;
    for (int i=0; i < hs; ++i) out_split[i] = 0.0f;
    for (int t2=0; t2 < T; ++t2) {
        const float* value_t2 = ip + b * T * C3 + t * T + h * hs + C * 2;
        for (int i=0; i < hs; ++i)
            out_split[i] += attention_bth[t2] * value_t2[i];
    }
}

void attention_forwad_gpu(
    float* out, 
    const float* inp, 
    int B, 
    int T, 
    int C,
    int NH,
    const int block_size
) {
    int total_threads = B * NH * T * T;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    float* preattention;
    cudaMalloc(&preattention, B * NH * T * T * sizeof(float));
    // Q dot K / √d
    attention_query_dot_key_kernel<<<num_blocks, block_size>>>(preattention, inp, B, T, C, NH);
    // softmax
    float* attention;
    cudaMalloc(&attention, B * NH * T * T * sizeof(float));
    total_threads = B * NH * T;
    num_blocks = (total_threads + block_size - 1) / block_size;
    attention_softmax_kernel<<<num_blocks, block_size>>>(attention, preattention, B, T, NH);
    // Attention dot V
    attention_output_kernel<<<num_blocks, block_size>>>(out, attention, inp, B, T, C, NH)
}

__global__ void forward_flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    const int B,
    const int T,
    const int D,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float* l,
    float* m,
    float* O,
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // gridDim.y = NH
    // gridDim.x = B
    int qkv_offset = (bx * gridDim.y * T * D) + (by * T * D); 
    int lm_offset = (bx * gridDim.y * T) + (by * T);

    // SRAM
    extern __shared__ float sram[];
    int Q_tile_size = Br * D;
    int KV_tile_size = Bc * D;
    float* Qi = sram;
    float* Kj = sram[Q_tile_size + KV_tile_size];
    float* Vj = sram[Q_tile_size + KV_tile_size * 2];

    for (int j < Tc; ++j) {
        for (int x = 0; x < D; ++x) {
            Kj[tx * D + x] = K[qkv_offset + (tile_size * j) + (tx * D + x)];
        }
    }
}


torch::Tensor forward_flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Q (B, NH, T, D)
    int M;
    cudaDeviceGetAttribute(&M, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    const int B = Q.size(0), NH = Q.size(1), T = Q.size(2), D = Q.size(3);

    // Bc = M / (4 * D)
    const int Bc = M / (4 * D);
    const int Br = min(M / (4 * D), D);
    const int Tr = ceil((float) T / Br);
    const int Tc = ceil((float) T / Bc);
    auto O = torch.zeros_like(Q);
    auto l = torch.zeros({B, NH, T});
    auto m = torch.full({B, NH, T}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);
    
    // Calculate Sram size needed per block
    const int sram_size = (2 * Bc * D + Br * D + Bc * Br) * sizeof(float);
    printf("Max shared memory per block: %d\n, requesting shared memory per block: %d\n", M, sram_size);

    //
    dim3 grid_dim(B, NH);
    dim3 block_dim(Br);

    forward_flash_attention<<<grid_dim, block_dim, sram_size>>>();
}