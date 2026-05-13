# MegaPairs Dataset Support

本文档说明了为 FlagEmbedding-MultiModal 添加的 MegaPairs 数据集支持。

## 概述

MegaPairs 是一个多模态图像检索数据集，包含查询图像、文本指令和目标图像的配对数据。此次更新为 FlagEmbedding-MultiModal 添加了完整的 MegaPairs 支持，包括：

- ✅ **评估支持**：可以在 MegaPairs 上评估多模态 embedder 和 reranker
- ✅ **训练支持**：可以直接使用 MegaPairs 格式数据进行训练
- ✅ **格式兼容**：训练和评估代码均支持 MegaPairs 原生格式

## MegaPairs 数据格式

### 原始格式
```json
{
  "q_img": "train-00100-to-00199/02280/022803773.jpg",
  "q_texts": [
    "Similar bathroom setup, but with a smaller mirror and a simpler frame.",
    "Replace the large illuminated mirror with a smaller, unlit one.",
    "Find a similar bathroom, but with a smaller mirror and a lighter-colored vanity."
  ],
  "t_img": "train-00100-to-00199/01574/015747381.jpg",
  "hns": [
    "train-00100-to-00199/02280/022803773.jpg",
    "train-00000-to-00099/03116/031168221.jpg",
    "train-00100-to-00199/03305/033052550.jpg",
    "train-00100-to-00199/01986/019869496.jpg",
    "train-00000-to-00099/00838/008383864.jpg",
    "train-00100-to-00199/03293/032935336.jpg"
  ]
}
```

### 字段说明
- `q_img`: 查询图像路径
- `q_texts`: 查询文本指令列表（描述检索意图）
- `t_img`: 目标图像路径（正样本）
- `hns`: 困难负样本图像路径列表（hard negatives）

## 新增文件结构

### 评估模块
```
FlagEmbedding/evaluation/mega_pairs/
├── __init__.py          # 模块导出
├── arguments.py         # 评估参数
├── data_loader.py       # 数据加载器
├── searcher.py          # 多模态检索和重排序器
├── evaluator.py         # 评估器
├── runner.py            # 运行器
└── __main__.py          # 主入口
```

### 评估脚本
```
examples/evaluation/mega_pairs/
├── eval_mega_pairs_embedder.sh   # Embedder 评估脚本
└── eval_mega_pairs_reranker.sh   # Reranker 评估脚本
```

### 训练脚本
```
examples/finetune/embedder/multimodal/
└── train_megapairs.sh   # Embedder 训练脚本

examples/finetune/reranker/multimodal/
└── train_megapairs.sh   # Reranker 训练脚本
```

## 核心功能实现

### 1. 数据加载器 (data_loader.py)

**功能**：从 HuggingFace 加载 MegaPairs 数据集并转换为标准评估格式

**转换逻辑**：
- **Corpus**：收集所有唯一图像（t_img + 所有 hns）
- **Queries**：为每个 q_text 创建一个 query（包含 text + image）
- **Qrels**：每个 query 的正确答案是对应的 t_img

**示例转换**：
```python
# 输入：1个MegaPairs样本
{
  "q_img": "img1.jpg",
  "q_texts": ["text1", "text2"],
  "t_img": "target.jpg",
  "hns": ["neg1.jpg", "neg2.jpg"]
}

# 输出：
# queries.jsonl
{"id": "0_0", "text": "text1", "image": "img1.jpg"}
{"id": "0_1", "text": "text2", "image": "img1.jpg"}

# corpus.jsonl
{"id": "target.jpg", "image": "target.jpg", "text": ""}
{"id": "neg1.jpg", "image": "neg1.jpg", "text": ""}
{"id": "neg2.jpg", "image": "neg2.jpg", "text": ""}

# qrels.jsonl
{"qid": "0_0", "docid": "target.jpg", "relevance": 1}
{"qid": "0_1", "docid": "target.jpg", "relevance": 1}
```

### 2. 多模态检索器 (searcher.py)

**MultimodalEvalRetriever**：
- 扩展 `EvalDenseRetriever` 以支持图像输入
- 自动检测 embedder 是否支持 `images` 参数
- 同时传递文本和图像给 embedder

**MultimodalEvalReranker**：
- 扩展 `EvalReranker` 以支持图像输入
- 支持 MultimodalReranker 的 `query_type` 和 `doc_type` 参数
- 处理图像到图像、文本到图像等多种检索场景

### 3. 训练数据支持

**修改的文件**：
- `FlagEmbedding/finetune/embedder/multimodal/base/dataset.py`
- `FlagEmbedding/finetune/reranker/multimodal/base/dataset.py`

**实现逻辑**：
```python
# 在 __getitem__ 方法中自动检测格式
if 'q_texts' in data and 'q_img' in data:
    # MegaPairs格式
    query_text = random.choice(data['q_texts'])  # 随机选一个
    query_image = data['q_img']
    pos_images = [data['t_img']]
    neg_images = data['hns']
else:
    # 原有格式
    query_text = data.get('query', ...)
    ...
```

**优势**：
- ✅ 向后兼容：仍支持原有训练数据格式
- ✅ 零转换：直接使用 MegaPairs 原生格式
- ✅ 统一接口：训练和评估使用相同的数据格式

## 使用方法

### 评估 Embedder

```bash
cd examples/evaluation/mega_pairs
bash eval_mega_pairs_embedder.sh
```

**支持的模型**：
- BAAI/BGE-VL-MLLM-S1
- 其他支持 multimodal-mllm 接口的模型

**关键参数**：
```bash
--embedder_name_or_path BAAI/BGE-VL-MLLM-S1
--embedder_model_class multimodal-mllm
--dataset_name JUNJIE99/MegaPairs
--splits test
```

### 评估 Reranker

```bash
cd examples/evaluation/mega_pairs
bash eval_mega_pairs_reranker.sh
```

**支持的模型**：
- jinaai/jina-reranker-m0
- 其他支持多模态的 reranker

**关键参数**：
```bash
--embedder_name_or_path BAAI/BGE-VL-MLLM-S1  # 用于初筛
--reranker_name_or_path jinaai/jina-reranker-m0  # 重排序
--search_top_k 1000
--rerank_top_k 100
```

### 训练 Embedder

```bash
cd examples/finetune/embedder/multimodal

# 修改脚本中的数据路径
# train_data="path/to/megapairs/train.jsonl"

bash train_megapairs.sh
```

**数据格式要求**：
- 直接使用 MegaPairs JSONL 格式
- 每行一个 JSON 对象，包含 q_img, q_texts, t_img, hns

### 训练 Reranker

```bash
cd examples/finetune/reranker/multimodal

# 修改脚本中的数据路径
# train_data="path/to/megapairs/train.jsonl"

bash train_megapairs.sh
```

## 评估指标

支持的评估指标：
- `ndcg_at_k`: Normalized Discounted Cumulative Gain
- `recall_at_k`: Recall @ K
- `map`: Mean Average Precision
- `mrr`: Mean Reciprocal Rank
- `precision_at_k`: Precision @ K

默认 k 值：`[1, 3, 5, 10, 20, 50, 100]`

## 技术细节

### 数据流程

**评估流程**：
1. 从 HuggingFace 加载 MegaPairs 数据集
2. 转换为标准格式（corpus.jsonl, queries.jsonl, qrels.jsonl）
3. MultimodalEvalRetriever 编码 queries 和 corpus
4. FAISS 索引进行检索
5. （可选）MultimodalEvalReranker 重排序
6. 计算评估指标

**训练流程**：
1. 加载 MegaPairs JSONL 文件
2. 自动检测格式（MegaPairs vs 原有格式）
3. 为每个样本随机选择一个 q_text
4. 采样正负样本进行对比学习
5. 使用 contrastive loss 或 knowledge distillation

### 兼容性设计

**自动格式检测**：
```python
if 'q_texts' in data and 'q_img' in data:
    # MegaPairs 格式处理
else:
    # 原有格式处理
```

**模型能力检测**：
```python
if 'images' in sig.parameters:
    # 支持多模态
    embedder.encode_corpus(texts, images=images)
else:
    # 仅文本
    embedder.encode_corpus(texts)
```

## 注意事项

⚠️ **未测试版本**：此次更新的代码尚未在服务器上测试，建议：
1. 先在小规模数据上测试
2. 检查图像路径是否正确
3. 确认模型加载正常
4. 验证评估指标是否合理

⚠️ **图像路径**：
- 默认使用数据集中的相对路径
- 可通过 `--image_root_dir` 指定图像根目录
- 确保图像文件可访问

⚠️ **内存使用**：
- Corpus 可能包含大量图像
- 建议使用 `--corpus_embd_save_dir` 缓存 embeddings
- 大规模评估时注意 GPU 内存

## 后续计划

- [ ] 在服务器上完整测试
- [ ] 优化大规模数据集的加载效率
- [ ] 添加更多评估指标
- [ ] 支持多语言 MegaPairs 变体
- [ ] 添加可视化分析工具

## 相关资源

- **MegaPairs 数据集**：[JUNJIE99/MegaPairs](https://huggingface.co/datasets/JUNJIE99/MegaPairs)
- **BGE-VL-MLLM-S1**：[BAAI/BGE-VL-MLLM-S1](https://huggingface.co/BAAI/BGE-VL-MLLM-S1)
- **Jina Reranker M0**：[jinaai/jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0)

## 贡献者

本次更新实现了完整的 MegaPairs 支持，包括评估和训练功能，保持了良好的向后兼容性。

