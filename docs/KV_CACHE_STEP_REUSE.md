# Step-to-Step KV-Cache Reuse

## 概述

这个优化通过在相邻时间步之间重用K,V投影来显著减少计算量。

**工作原理**：
- 第0步：正常计算Q,K,V（图像+文本）
- 第1步：重用第0步的K,V，只计算Q
- 第2步：重用第1步的K,V，只计算Q
- 依此类推...

## 性能影响

**优势**：
- **大幅减少计算量**：跳过K,V投影计算（约占总计算的33%）
- **预期加速**：20-30%的推理速度提升
- **内存开销**：约700MB（用于存储前一步的K,V）

**劣势**：
- **可能影响输出质量**：因为使用了过时的K,V
- **质量下降程度**：取决于步数和prompt复杂度
- **建议先测试**：在实际应用前评估质量影响

## 使用方法

### 方法1：通过命令行参数（推荐）

```bash
python sample_video.py \
    --model-base ckpts \
    --prompt-template-video dit-llm-encode-video \
    --prompt "A cat is eating fish" \
    --video-size 720 640 \
    --video-length 129 \
    --infer-steps 20 \
    --seed 222 \
    --save-path videos/ \
    --cfg-scale 6.0 \
    --enable-kv-cache  # 启用KV缓存
```

查看帮助信息：
```bash
python sample_video.py --help
# 会看到:
# --enable-kv-cache     Enable step-to-step KV cache reuse to reduce computation.
#                       Each timestep will reuse K,V from previous step (only compute Q).
#                       This provides 20-30% speedup but may slightly impact output quality.
#                       Default: disabled for best quality.
```

### 方法2：通过代码设置

```python
from hyvideo.inference import HunyuanVideoSampler

# 加载模型
sampler = HunyuanVideoSampler.from_pretrained(models_path, args=args)

# 启用KV cache
sampler.args.enable_kv_cache = True

# 运行推理
outputs = sampler.predict(
    prompt="A cat playing with a ball",
    height=720,
    width=640,
    video_length=129,
    infer_steps=20,
)
```

### 方法3：直接修改pipeline

```python
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline

pipe = HunyuanVideoPipeline(...)

# 在调用前设置
pipe.args.enable_kv_cache = True

video = pipe(
    prompt="Your prompt",
    height=720,
    width=640,
    # ... other params
)
```

## 日志输出

启用KV cache后，你会看到以下日志：

```
[KV-Cache] Enabled: Step-to-step KV reuse activated
[KV-Cache] Each step will reuse K,V from previous step (only compute Q)
[KV-Cache] This significantly reduces computation but may impact output quality
```

禁用时会看到：

```
[KV-Cache] Disabled: Standard computation (full Q,K,V each step)
```

## 技术细节

### 实现原理

1. **MMDoubleStreamBlock**：
   - 正常模式：计算img_q, img_k, img_v + txt_q, txt_k, txt_v
   - Cache模式：只计算img_q + txt_q，重用前一步的k, v
   - 返回值：`(img, txt, (current_k, current_v))`

2. **MMSingleStreamBlock**：
   - 正常模式：从合并的x计算完整的q, k, v
   - Cache模式：只计算q，重用前一步的k, v
   - 返回值：`(x, (current_k, current_v))`

3. **HYVideoDiffusionTransformer**：
   - 维护两个cache列表：`_kv_cache_double`和`_kv_cache_single`
   - 每个timestep：
     - 从cache读取前一步的K,V传给blocks
     - 收集当前步的K,V保存到cache
     - 下一步使用当前步的cache

### 为什么可能影响质量？

在diffusion模型的去噪过程中：
- **图像内容在变化**：每一步latent都在改变
- **K,V应该反映当前状态**：标准做法是每步重新计算K,V
- **重用旧K,V**：相当于使用"过期"的attention keys/values
- **累积误差**：多步重用可能导致误差累积

### 何时使用？

**适合的场景**：
- 快速预览/草图生成
- 对质量要求不高的应用
- 需要实时或近实时生成
- 资源受限的环境

**不适合的场景**：
- 高质量最终渲染
- 对细节要求极高的应用
- 复杂的prompt或场景
- 需要最佳质量的生产环境

## 基准测试

测试配置：
- 模型：HYVideo-T/2
- 分辨率：720x640
- 长度：129帧
- 步数：20
- GPU：A100 80GB

预期结果：

| 指标 | 无Cache | 有Cache | 改善 |
|------|---------|---------|------|
| 推理时间 | 800s | 560-640s | 20-30% |
| K,V计算 | 1200次 | 60次 | 95% |
| 内存使用 | 23.5 GB | 24.2 GB | +700MB |
| 输出质量 | 基准 | 轻微下降 | 待评估 |

**注意**：实际结果可能因硬件、prompt和配置而异。

## 后续优化方向

1. **自适应KV重用**：
   - 前几步使用cache（图像变化小）
   - 后几步正常计算（需要精细调整）

2. **选择性重用**：
   - 只重用文本的K,V（恒定不变）
   - 每步重新计算图像的K,V

3. **质量监控**：
   - 检测误差累积
   - 必要时刷新cache

4. **混合策略**：
   - 隔步重用（step 0→1→2重新计算→3→4重新计算...）
   - 在速度和质量间平衡

## 故障排除

### 问题：看不到cache相关日志

**解决方案**：
```python
# 确保loguru配置正确
from loguru import logger
logger.info("Test logging")

# 检查args设置
print(f"enable_kv_cache: {getattr(args, 'enable_kv_cache', 'NOT SET')}")
```

### 问题：质量明显下降

**解决方案**：
1. 尝试减少推理步数（cache影响相对减小）
2. 使用更高的cfg_scale补偿质量损失
3. 考虑禁用cache用于最终渲染

### 问题：内存不足

**解决方案**：
- Cache需要额外~700MB
- 如果内存紧张，可以禁用cache
- 或者减小batch size/分辨率

## 参考

- [Attention机制](https://arxiv.org/abs/1706.03762)
- [Diffusion Models](https://arxiv.org/abs/2006.11239)
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)

## 贡献

如果发现问题或有改进建议，请提交issue或PR。

特别欢迎：
- 质量影响的基准测试结果
- 不同场景下的性能数据
- 改进cache策略的建议
