# 多模态Reranker适配指南

## 📋 完成的工作

成功将 `jinaai/jina-reranker-m0` 多模态reranker适配到FlagEmbedding框架，支持文本和图像的各种组合重排序。

## 🗂️ 创建的文件列表

### 1. 核心代码 (2个文件)

| 文件 | 说明 |
|------|------|
| `FlagEmbedding/inference/reranker/multimodal/__init__.py` | 模块导出 |
| `FlagEmbedding/inference/reranker/multimodal/base.py` | ⭐ 多模态reranker实现 |

### 2. 更新的文件 (3个文件)

| 文件 | 修改内容 |
|------|---------|
| `FlagEmbedding/inference/reranker/model_mapping.py` | 添加jina-reranker-m0映射 |
| `FlagEmbedding/inference/reranker/__init__.py` | 导出MultimodalReranker |
| `FlagEmbedding/inference/__init__.py` | 导出到顶层 |

### 3. 测试脚本 (3个文件)

| 文件 | 说明 |
|------|------|
| `examples/inference/reranker/multimodal/auto_reranker_single_device.py` | 使用FlagAutoReranker |
| `examples/inference/reranker/multimodal/base_reranker_single_device.py` | 直接使用MultimodalReranker |
| `examples/inference/reranker/multimodal/README.md` | 详细使用文档 |

## 🎯 支持的模态组合

| Query类型 | Document类型 | 用途 |
|----------|-------------|------|
| Text | Image | 文本查询图像 |
| Image | Text | 图像查询文本 |
| Text | Text | 文本查询文本 |
| Image | Image | 图像查询图像 |

## 🔑 核心技术实现

### 1. **多模态输入处理**

```python
def formatting_prompts_func(query, doc, query_type='text', doc_type='text'):
    if query_type == 'image':
        query_part = "**Query**:\n<|vision_start|><|image_pad|><|vision_end|>"
    else:
        query_part = f"**Query**:\n{query}"
    
    if doc_type == 'image':
        doc_part = "**Document**:\n<|vision_start|><|image_pad|><|vision_end|>"
    else:
        doc_part = f"**Document**:\n{doc}"
    
    return doc_part + '\n' + query_part
```

### 2. **图像加载**

支持多种图像源：
- 本地文件路径
- URL地址
- PIL Image对象

```python
def load_images(images):
    images_batch = []
    for image in images:
        if isinstance(image, Image.Image):
            images_batch.append(image)
        else:
            pil_image = load_image(image)  # 支持URL和本地路径
            images_batch.append(pil_image)
    return images_batch
```

### 3. **评分机制**

```python
# Forward通过模型
scores = self.model(**batch).view(-1).cpu().float().numpy()

# Sigmoid归一化到[0,1]
if normalize:
    scores = 1.0 / (1.0 + np.exp(-(scores - LOGIT_BIAS)))
```

### 4. **特殊Token处理**

```python
# 添加score token用于计算相关性
batch["input_ids"] = torch.cat([
    batch["input_ids"],
    torch.full((batch_size, 1), self.score_token_id, device=...),
], dim=1)
```

## 🚀 使用方法

### 方法1：使用FlagAutoReranker（推荐）

```python
from FlagEmbedding import FlagAutoReranker

reranker = FlagAutoReranker.from_finetuned(
    'jinaai/jina-reranker-m0',
    devices="cuda:0"
)

# Text query -> Image documents
query = "machine learning paper"
docs = [
    "https://example.com/paper1.png",
    "https://example.com/paper2.png"
]

pairs = [[query, doc] for doc in docs]
scores = reranker.compute_score(
    pairs,
    query_type='text',
    doc_type='image'
)

print(scores)  # [0.789, 0.456]
```

### 方法2：直接使用MultimodalReranker

```python
from FlagEmbedding import MultimodalReranker

reranker = MultimodalReranker(
    model_name_or_path='jinaai/jina-reranker-m0',
    use_fp16=True,
    devices="cuda:0",
    batch_size=8,
    max_length=2048,
    normalize=True,
    query_type='text',    # 默认query类型
    doc_type='image',     # 默认document类型
)

scores = reranker.compute_score(
    sentence_pairs,
    query_type='text',
    doc_type='image'
)
```

## 📊 与原始实现的对比

### 原始实现 (raw/inference_jina_reranker_m0.py)

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    'jinaai/jina-reranker-m0',
    trust_remote_code=True
)
model.to('cuda')

image_pairs = [[query, doc] for doc in documents]
scores = model.compute_score(
    image_pairs, 
    max_length=2048, 
    doc_type="image"
)
```

### 适配后的实现

```python
from FlagEmbedding import FlagAutoReranker

reranker = FlagAutoReranker.from_finetuned('jinaai/jina-reranker-m0')
scores = reranker.compute_score(
    image_pairs, 
    query_type='text',
    doc_type='image'
)
```

**优势**：
- ✅ 统一的API接口
- ✅ 自动模型映射
- ✅ 支持多GPU
- ✅ 批处理优化

## 🔧 参数说明

### 初始化参数

```python
MultimodalReranker(
    model_name_or_path='jinaai/jina-reranker-m0',
    use_fp16=True,              # 使用半精度
    trust_remote_code=True,     # 必须为True
    devices="cuda:0",            # 设备
    batch_size=8,               # 批大小
    query_max_length=512,       # 查询最大长度
    max_length=10240,           # 总最大长度
    normalize=True,             # 归一化分数
    query_type='text',          # 默认query类型
    doc_type='image',           # 默认doc类型
)
```

### 计算分数参数

```python
scores = reranker.compute_score(
    sentence_pairs,              # [(query, doc), ...]
    batch_size=8,               # 可选：覆盖批大小
    query_max_length=512,       # 可选：查询最大长度
    max_length=10240,           # 可选：总最大长度
    normalize=True,             # 可选：是否归一化
    query_type='text',          # 'text' 或 'image'
    doc_type='image',           # 'text' 或 'image'
)
```

## 📝 使用示例

### 示例1：文本查询→图像文档

```python
reranker = FlagAutoReranker.from_finetuned('jinaai/jina-reranker-m0')

query = "scientific research paper"
image_docs = [
    "https://example.com/paper1.png",
    "https://example.com/article.png",
    "https://example.com/cat.png"
]

pairs = [[query, doc] for doc in image_docs]
scores = reranker.compute_score(
    pairs,
    query_type='text',
    doc_type='image'
)

# 排序
sorted_results = sorted(zip(image_docs, scores), key=lambda x: x[1], reverse=True)
print(f"Top result: {sorted_results[0][0]} (score: {sorted_results[0][1]:.4f})")
```

### 示例2：图像查询→文本文档

```python
image_query = "path/to/query_image.png"
text_docs = [
    "This paper discusses deep learning architectures.",
    "A recipe for chocolate cake.",
    "Machine learning research overview."
]

pairs = [[image_query, doc] for doc in text_docs]
scores = reranker.compute_score(
    pairs,
    query_type='image',
    doc_type='text'
)
```

### 示例3：混合批次处理

```python
# 处理不同类型的pairs
mixed_pairs = [
    ("text query", "text document"),           # text-text
    ("text query", "image.png"),              # text-image
    ("query.png", "text document"),           # image-text
    ("query.png", "doc.png"),                 # image-image
]

# 需要分别处理不同类型
text_text_scores = reranker.compute_score(
    [mixed_pairs[0]],
    query_type='text', doc_type='text'
)
text_image_scores = reranker.compute_score(
    [mixed_pairs[1]],
    query_type='text', doc_type='image'
)
```

## 🎯 核心特性

### 1. 自动模型映射

在 `model_mapping.py` 中配置：
```python
(
    "jinaai/jina-reranker-m0",
    RerankerConfig(MultimodalReranker, trust_remote_code=True)
),
```

自动识别并加载对应的reranker类。

### 2. 处理器集成

```python
# 自动加载processor
self.processor = AutoProcessor.from_pretrained(
    model_name_or_path,
    max_pixels=602112,
    min_pixels=3136,
    trust_remote_code=True
)
```

### 3. 分数归一化

```python
# LOGIT_BIAS = 2.65
scores = 1.0 / (1.0 + np.exp(-(raw_scores - LOGIT_BIAS)))
```

将原始分数映射到 [0, 1] 区间。

## 🐛 故障排查

### 1. ImportError: trust_remote_code
**解决**：确保设置 `trust_remote_code=True`

### 2. 图像加载失败
**解决**：
- 检查路径/URL是否正确
- 确保图像格式支持（PNG, JPG, WEBP）
- 检查网络连接（如果是URL）

### 3. OOM错误
**解决**：
- 减小 `batch_size`
- 启用 `use_fp16=True`
- 减小 `max_length`

### 4. 分数异常
**解决**：
- 检查 `query_type` 和 `doc_type` 是否正确
- 确认 `normalize=True` 以获得 [0,1] 分数

## 📦 模型信息

**jina-reranker-m0**:
- 基础架构：Qwen2-VL
- 支持输入：文本和图像
- 输出：相关性分数 [0, 1]（归一化后）
- 最大序列长度：10240 tokens
- 特殊token：score_token_id = 100

## 🔄 执行流程

```
1. 加载模型和processor
   └─> AutoModel.from_pretrained()
   └─> AutoProcessor.from_pretrained()

2. 格式化prompt
   └─> formatting_prompts_func()

3. 加载图像（如果需要）
   └─> load_images()

4. 处理输入
   └─> processor(text=..., images=...)

5. 添加score token
   └─> append score_token_id

6. 前向传播
   └─> model.forward()

7. 归一化分数
   └─> sigmoid normalization
```

## ✅ 总结

本次适配实现了：
1. ✅ 完整的多模态reranker支持
2. ✅ 支持4种模态组合（text-text, text-image, image-text, image-image）
3. ✅ 统一的API接口
4. ✅ 自动模型映射
5. ✅ 多GPU支持
6. ✅ 批处理优化
7. ✅ 分数归一化

现在可以使用统一的FlagEmbedding框架来使用jina-reranker-m0了！🎉

