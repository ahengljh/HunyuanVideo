# KV Cache Reuse for HunyuanVideo

## 概述

KV Cache重用是一种优化DiT视频生成推理过程的创新技术。在视频生成过程中，许多区域的变化会逐渐收敛，前后帧之间的差异会变得非常微小。通过检测这些变化较小的区域，我们可以重用之前计算的KV cache，而不是重新计算，从而节省大量计算资源和内存。

## 核心思想

1. **区域变化检测**：在每个去噪步骤之间，计算latents在空间-时间patch上的变化量（delta）
2. **选择性重用**：当某个patch的变化小于设定阈值时，重用该patch对应的KV cache
3. **动态优化**：随着去噪过程的推进，越来越多的区域会收敛，重用率会逐渐提高

## 使用方法

### 命令行参数

```bash
python sample_video.py \
    --prompt "一只可爱的熊猫在竹林里玩耍" \
    --enable-kv-cache \
    --kv-cache-threshold 0.1 \
    --kv-cache-patch-size 1 2 2 \
    --kv-cache-aggregation l2
```

### 参数说明

- `--enable-kv-cache`: 启用KV cache重用机制
- `--kv-cache-threshold`: 变化阈值（默认0.1）
  - 较低的值（0.05-0.08）：更严格的重用条件，质量更高但重用率较低
  - 中等的值（0.1-0.15）：平衡质量和速度
  - 较高的值（0.15-0.2）：更激进的重用，速度更快但可能影响质量
- `--kv-cache-patch-size`: Patch大小 [T, H, W]（默认[1, 2, 2]）
  - 较小的patch：更细粒度的控制，但计算开销稍大
  - 较大的patch：更粗粒度，计算开销小，但可能过度重用
- `--kv-cache-aggregation`: 变化度量方法（默认"l2"）
  - `l2`: L2范数归一化距离
  - `cosine`: 余弦距离
  - `mse`: 均方误差

## 示例脚本

### 基础示例

```bash
# 使用默认参数启用KV cache
python sample_video.py \
    --prompt "A beautiful sunset over the ocean" \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --enable-kv-cache
```

### 高质量模式

```bash
# 使用较低阈值，追求更高质量
python sample_video.py \
    --prompt "A cat playing with a ball of yarn" \
    --enable-kv-cache \
    --kv-cache-threshold 0.05 \
    --kv-cache-patch-size 1 2 2
```

### 快速模式

```bash
# 使用较高阈值，追求更快速度
python sample_video.py \
    --prompt "A dog running in the park" \
    --enable-kv-cache \
    --kv-cache-threshold 0.15 \
    --kv-cache-patch-size 2 4 4
```

## 性能特性

### 预期效果

1. **计算节省**：
   - 早期步骤（1-20步）：重用率较低（~10-30%）
   - 中期步骤（20-35步）：重用率逐渐提高（~30-60%）
   - 后期步骤（35-50步）：重用率较高（~60-85%）

2. **内存影响**：
   - 每个block需要额外存储上一步的K和V张量
   - 内存开销相对较小（约增加5-10%）

3. **质量影响**：
   - 使用合理的阈值（0.08-0.12），质量影响minimal
   - 过高的阈值可能导致细节丢失或运动不自然

### 日志输出

启用KV cache后，您会看到类似以下的日志输出：

```
INFO - KV cache enabled with threshold=0.1, patch_size=(1, 2, 2), aggregation=l2
INFO - Step 10: KV cache reuse ratio = 25.34%
INFO - Step 20: KV cache reuse ratio = 45.67%
INFO - Step 30: KV cache reuse ratio = 62.89%
INFO - Step 40: KV cache reuse ratio = 78.12%
INFO - KV cache statistics:
INFO -   Average reuse ratio: 56.78%
INFO -   Total patches: 245760
INFO -   Total reused: 139567
INFO -   Total recomputed: 106193
```

## 技术细节

### 实现架构

1. **Delta计算** (`hyvideo/modules/kv_cache_utils.py`)
   - `compute_latent_delta_mask()`: 计算patch-wise的变化mask
   - 支持多种度量方法（L2, cosine, MSE）

2. **KV存储** (`hyvideo/modules/models.py`)
   - `MMDoubleStreamBlock`: 为图像流存储KV cache
   - `MMSingleStreamBlock`: 为合并流存储KV cache

3. **Pipeline集成** (`hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py`)
   - 在去噪循环中计算delta mask
   - 将mask传递给transformer
   - 收集和报告统计信息

### 关键函数

```python
# 计算变化mask
mask = compute_latent_delta_mask(
    current_latents,    # 当前步骤的latents
    previous_latents,   # 上一步骤的latents
    threshold=0.1,      # 阈值
    patch_size=(1,2,2), # Patch大小
    aggregation="l2"    # 度量方法
)

# 应用KV cache重用
k_updated, v_updated = apply_kv_cache_mask(
    new_k, new_v,       # 新计算的K和V
    cached_k, cached_v, # 缓存的K和V
    reuse_mask,         # 重用mask
    img_len             # 图像序列长度
)
```

## 最佳实践

1. **阈值选择**：
   - 从默认值0.1开始
   - 如果发现质量下降，降低到0.05-0.08
   - 如果想要更快速度且能接受轻微质量损失，提高到0.15-0.2

2. **Patch大小**：
   - 对于高分辨率视频（720p+），使用[1, 2, 2]或[1, 4, 4]
   - 对于低分辨率视频，使用[1, 2, 2]
   - 时间维度通常保持为1以保持时间连贯性

3. **度量方法**：
   - `l2`: 适合大多数情况，平衡准确性和速度
   - `cosine`: 对光照变化不敏感，适合有光照变化的场景
   - `mse`: 计算稍快，但对异常值敏感

## 限制和注意事项

1. **不适用场景**：
   - 需要极高质量的艺术创作
   - 快速运动或剧烈变化的场景
   - 推理步数较少（<30步）的情况

2. **兼容性**：
   - 与classifier-free guidance兼容
   - 与embedded guidance兼容
   - 目前不支持与某些高级并行策略（Ring/Ulysses attention）同时使用

3. **调试**：
   - 查看日志中的重用率统计
   - 如果重用率异常低（<20%），可能需要调高阈值
   - 如果重用率异常高（>90%），可能需要降低阈值

## 贡献者

此功能由Claude基于用户的创意想法实现。

## 许可证

遵循HunyuanVideo项目的原始许可证。
