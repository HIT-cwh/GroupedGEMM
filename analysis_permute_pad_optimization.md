# MOE Permute Pad/Unpad 优化分析（v3 修正版）

**文件:** `csrc/permute_128x.cu` (2457 行)
**分析范围:** `moe_permute_topK_op_pad` (前向), `moe_recover_topK_op_unpad` (unpermute), `moe_recover_topK_unpad_bwd_op` (反向)
**约束:** per-expert pad 行和 UB tail 均需要清零

---

## 一、当前实现流程

### Forward: `moe_permute_topK_op_pad` (行 985-1348)

```
步骤1:  CUB RadixSort          — 按 expert_id 排序 token indices
步骤2:  CUB HistogramEven      — 统计每个 expert 的 token 数 → expert_counts[E]
步骤3:  compute_padded_offsets  — <<<1,1>>> 单线程前缀和，计算 padded_offsets[E+1]
步骤4:  计算 UB 上界           — UB = ceil(M/128)*128 + 128*E（CPU 端，无 GPU sync）
步骤5:  build_padded_sorted_row_id — <<<E,256>>> 每 expert 构建 padded 布局
步骤6:  cudaMemsetAsync         — row_id_map 初始化为 -1
步骤7:  scatter_row_id_map     — <<<ceil(UB/256),256>>> 构建 row_id_map
步骤8:  moe_permute_topK_kernel — <<<num_tokens, threads>>> 数据搬运（scatter 写法）
步骤9:  zero_pad_by_segments   — <<<dim3(E,128), threads>>> 对 pad 行写零
步骤10: zero_ub_tail           — <<<ceil(UB*vec_cols/256), 256>>> 对 UB 尾部写零
```

---

## 二、问题分析

---

### 问题 #1: `zero_ub_tail_kernel` — 巨量空 block 调度 🔴 P0

**行号:** 216-247 (kernel), 1152-1165 等 (调用处)

#### 问题根因

UB tail 范围 = `[padded_offsets[E], UB)`。 `padded_offsets[E]` 在 CPU 端未知（为避免 D2H sync），所以 grid 按整个 UB buffer 大小计算：

```cpp
int64_t total_vec_ub = (int64_t)num_out_tokens_ub * vec_cols;   // UB × (cols/kEPA)
int blocks_tail = (int)((total_vec_ub + zthreads - 1) / zthreads);
```

kernel 内部通过 `padded_offsets[E]` 在 GPU 端判断边界，大量 block 发现自己负责的区域在 `offsets[E]` 之前，立即返回。

#### 量化分析（BF16, h=7168, M=65536, E=256）

```
UB          = 65536 + 32768 = 98304
vec_cols    = 7168 / 8 = 896
grid blocks = ceil(98304 × 896 / 256) = 344,064

实际 tail 行数 = UB - padded_offsets[E]
  均匀路由: count=256/expert → pad=0 → tail = 98304-65536 = 32768 行 → 需要 114,688 blocks
  不均匀:   tail 更小

空闲 blocks = 344,064 - 114,688 = 229,376 (均匀路由下 ~67% 空闲)
```

而且这里有一个更根本的问题：**grid 的索引基准是从 output[0] 开始的**（行 235-238），即每个 thread 算一个全局 idx 再判断是否落在 tail 范围内。这意味着所有 `idx < tail_start * vec_cols` 的 thread 都是废的。

#### 方案对比

| 方案 | 原理 | Grid 大小 | 缺点 |
|------|------|----------|------|
| **当前** | 全 UB 范围 flat launch，kernel 内判断 | ceil(UB × vec_cols / 256) = **344K** | 海量空 block |
| **A: tail 上界 grid** | 改基准：只启动覆盖 [UB-128E, UB) 的 block | ceil(128E × vec_cols / 256) = **114K** | 仍有空闲 block（实际 tail < 128E） |
| **B: 合并进 pad kernel** | 见问题 #2 方案 | **E+1** | tail 段可能行数多，单 block 慢（见下分析） |
| **C: 偏移基准 launch** | grid 只覆盖 max tail 行数，kernel 内 `base = offsets[E]` | ceil(128E × vec_cols / 256) = **114K** | 同 A，但基准对了 |
| **D: 小 kernel 写 grid 参数 + indirect launch** | <<<1,1>>> 读 offsets[E] 算精确 grid 写入 device buffer，再 indirect launch | **精确** | 需要 CUDA ≥12 或 device launch |
| **E: cudaMemsetAsync 只清 tail 的上界范围** | 对 output 尾部 128E 行做 memset | N/A (DMA) | 多清了一些（实际 tail < 128E） |

**最实际的优化：方案 E — 对 tail 区域用 cudaMemsetAsync**

tail 的上界行数 = `128*E`（CPU 端已知），真实 tail ≤ 这个值。我们在 output 的最后 128*E 行做 memset：

```cpp
// output 总行数 = UB = M_rounded + 128*E
// 最后 128*E 行一定包含了全部 tail + 部分有效数据（被 permute 覆写无影响）
// 但注意：这 128*E 行中也包含了部分 expert 的有效数据行！
```

**等等，不能这么做。** 因为 `UB - 128*E = ceil(M/128)*128`，而 padded_offsets[E] 的值取决于路由。如果某些 expert 得到很多 token，padded_offsets[E] 可能 > UB - 128*E，这时 memset 最后 128E 行会覆盖有效数据。

**修正方案 E'：在 permute kernel 之前做 memset，permute 再覆写**

不行——之前已分析过，全量 memset 浪费带宽，部分 memset 又可能覆盖有效数据（时序问题：permute 写和 memset 写并发不安全）。

**实际最优：方案 C — 偏移基准 flat launch**

重写 `zero_ub_tail_kernel`，将索引基准从 output[0] 改为从 `offsets[E]` 开始计数：

```cuda
template <typename T, int kElementsPerAccess>
static __global__ void zero_ub_tail_kernel_v2(
    T* __restrict__ output,
    const int* __restrict__ offsets,  // [E+1]
    int E,
    int num_cols,
    int ub_total,
    int max_tail_rows)  // CPU 传入 = 128*E (或 127*E)
{
    int64_t num_cols_i64 = (int64_t)num_cols;
    int tail_start = offsets[E];

    // 此 thread 负责 tail 中的第 idx 个向量
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t max_tail_vec = (int64_t)max_tail_rows * (num_cols_i64 / kElementsPerAccess);
    if (idx >= max_tail_vec) return;

    // 换算为绝对位置
    int64_t abs_elem = (int64_t)tail_start * num_cols_i64 + idx * kElementsPerAccess;

    // 边界检查：不超出 UB 总长
    if (abs_elem + kElementsPerAccess > (int64_t)ub_total * num_cols_i64) return;

    T* dst = output + abs_elem;

    using Frag = cutlass::Array<T, kElementsPerAccess>;
    Frag z;
    #pragma unroll
    for (int i = 0; i < kElementsPerAccess; ++i) z[i] = T(0);
    *(float4*)dst = *(float4*)(z.data());
}
```

**grid = ceil(128*E × vec_cols / 256)**

量化对比（BF16, h=7168, E=256）：
```
当前:   grid = ceil(98304 × 896 / 256) = 344,064 blocks
方案C:  grid = ceil(32768 × 896 / 256) = 114,688 blocks
                                           ↓ 减少 67%
```

而 114,688 blocks 中，有多少真正干活取决于实际 tail 大小。
均匀路由下 tail = 32768 行 → 几乎全部干活，很好。
极度不均匀下 tail 很小 → 大部分空闲，但总 block 数已经不算太离谱。

**进一步优化：如果 E 较大（比如 512），128*E 已经 65536 行**

这时方案 C 的 grid 还是很大。可以考虑**方案 B+C 混合**：用一个 kernel 同时处理 pad 和 tail，但 tail 部分分配多个 block 而不是 1 个。见问题 #2 中的统一方案。

---

### 问题 #2: `zero_pad_by_segments_kernel` — 固定 grid.y=128 🟠 P1

**行号:** 174-212 (kernel), 1142-1149 等 (调用处)

#### 问题

`dim3(E, 128)` = E×128 blocks。每个 expert pad ≤127 行，但实际经常远少于此。

#### 推荐方案：<<<E, threads>>> + block 内循环

```cuda
template <typename T, int kElementsPerAccess>
static __global__ void zero_pad_by_segments_kernel_v2(
    T* __restrict__ output,
    const int* __restrict__ counts,
    const int* __restrict__ offsets,
    int E, int num_cols)
{
    using Frag = cutlass::Array<T, kElementsPerAccess>;
    int e = blockIdx.x;
    if (e >= E) return;

    int pad_start = offsets[e] + counts[e];
    int pad_rows  = offsets[e + 1] - pad_start;
    if (pad_rows <= 0) return;

    int tid = threadIdx.x;
    int64_t num_cols_i64 = (int64_t)num_cols;
    Frag z;
    #pragma unroll
    for (int i = 0; i < kElementsPerAccess; ++i) z[i] = T(0);

    for (int r = 0; r < pad_rows; ++r) {
        T* row_ptr = output + (int64_t)(pad_start + r) * num_cols_i64;
        for (int c = tid * kElementsPerAccess; c < num_cols;
             c += blockDim.x * kElementsPerAccess) {
            *(float4*)(row_ptr + c) = *(float4*)(z.data());
        }
    }
}
```

Grid: **E blocks**（vs E×128）。减少 **128 倍**。

每 block 工作量：最多 127 行 × h 列。以 BF16 h=7168 为例：
- 127 × 7168 × 2B = 1.78 MB
- 256 threads × 16B/write = 4KB/轮 → ~445 轮
- E 个 block 并行，SM 充足情况下瞬间完成

---

### 进一步考虑：pad + tail 合并为一个 kernel？

你提到的担心很对：如果把 tail 也塞进一个 block 内循环，tail 可能有 32K 行，单 block 循环太慢会成为 straggler。

**两种策略：**

**策略 1: 保持两个 kernel，各自优化（推荐）**

```
zero_pad_by_segments_kernel_v2  <<<E, threads>>>           // pad: 最多 127 行/block
zero_ub_tail_kernel_v2          <<<ceil(128E*vc/256), 256>>> // tail: flat launch，基准偏移
```

简单、清晰。pad kernel 的 E 个 block 都很轻。tail kernel 的 block 都做实际写入（或接近）。两个 kernel launch 开销约 ~10 µs 可接受。

**策略 2: 合并为一个 kernel，tail 用多 block**

```cuda
// grid = E + num_tail_blocks
// blockIdx < E  → 处理 expert pad 行（内循环 ≤127 行）
// blockIdx >= E → 处理 tail 行（每 block 一行或多行）
```

省 1 次 launch，但 block 间工作量不均匀——前 E 个 block ≤127 行，后面的 block 可能只 1 行。不如策略 1 清晰，收益也不大。

**结论：推荐策略 1。**

---

### 问题 #3: `build_padded_sorted_row_id_kernel` — O(E²) 串行前缀和 🟡 P2

**行号:** 63-87

每个 expert block 串行累加 `for(i=0;i<e;i++) in_start += counts[i]`。E=256 → 32K 次冗余加法。

**修复：** 扩展 `compute_padded_offsets_kernel` 顺便输出 `count_offsets[E+1]`:

```cuda
// 在 compute_padded_offsets_kernel 循环中增加:
count_offsets[0] = 0;
for (int e = 0; e < E; ++e) {
    count_offsets[e + 1] = count_offsets[e] + counts[e];  // 新增
    int pc = round_up_int(counts[e], align);
    padded_counts[e] = pc;
    padded_offsets[e + 1] = padded_offsets[e] + pc;
}

// build_padded_sorted_row_id_kernel 中:
// 旧: int in_start = 0; for (int i = 0; i < e; ++i) in_start += counts[i];
// 新: int in_start = count_offsets[e];
```

零额外 kernel launch，O(E²) → O(E)。

---

### 问题 #4: UB 上界过度分配 🟡 P2

**行号:** 1077-1078

`UB = ceil(M/128)*128 + 128*E`，但每个 expert 最大 pad = 127。
改为 `M + 127*E` 更紧。

---

### 问题 #5: Unpermute 中 BF16 vs 其他类型不一致 🟠 P1

**行号:** 1460-1619

BF16 用 `moe_recover_topK_kernel_nopad_prob`，其他类型走旧 launcher。功能一致但维护隐患。
**推荐：** 统一。

---

### 问题 #6: 每次调用临时 Tensor 分配 🟡 P2

6 个临时 tensor 每次 forward 重新分配，合计 ~6-18 µs。
**推荐：** 缓存到 workspace。

---

### 问题 #7: switch-case 代码重复 🟢 P3

~800 行重复，模板 dispatch 可缩减。低优先级。

---

## 三、优化路线图

### Phase 1: 核心优化（< 1 天）

| # | 改动 | 效果 |
|---|------|------|
| **1** | `zero_ub_tail_kernel` 重写：索引基准从 output[0] 改为 offsets[E]，grid 按 128*E 上界 | Grid 344K → 115K blocks（减 67%），消除无效 block |
| **2** | `zero_pad_by_segments_kernel` 改 <<<E>>> + 内循环 | Grid E×128 → E blocks（减 128 倍） |
| **3** | `compute_padded_offsets` 输出 count_offsets | O(E²) → O(E) |
| **4** | UB 公式改 `M + 127*E` | 省 E 行显存 |

### Phase 2

| # | 改动 | 效果 |
|---|------|------|
| **5** | 统一 unpermute kernel | 代码一致性 |
| **6** | 临时 tensor 缓存 | 省 ~10 µs/call |

---

## 四、量化收益（BF16, h=7168, M=65536, E=256）

| 指标 | 优化前 | Phase 1 后 |
|------|--------|-----------|
| pad kernel grid | E×128 = 32,768 | **E = 256** |
| tail kernel grid | 344,064 | **114,688** |
| tail kernel 空闲 block 率 | ~67% (均匀) ~99% (偏斜) | **~0% (均匀) ~降低** |
| 总 zero kernel blocks | ~377K | **~115K** |
| build_padded 复杂度 | O(E²) | **O(E)** |
| kernel launches (fwd) | 10 | **10（不变）** |
| 额外显存浪费 | 128*E 行 | **127*E 行** |
