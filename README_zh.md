# FlagEmbedding-MultiModal  
> FlagEmbedding 多模态扩展（非官方）

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-≥3.8-green.svg)

## 声明
本仓库是由 [@86MaxCao](https://github.com/86MaxCao) 维护的**个人 fork**，
与 [FlagOpen](https://github.com/FlagOpen) **无关联**。  
所有原始代码遵循 Apache 2.0 协议；详见 [LICENSE](LICENSE)。

---

## 多模态向量与重排

本 fork 为 [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) 扩展了多模态向量和重排能力，提供统一的推理与 LoRA 微调 API，支持多种视觉语言模型，完全兼容 **transformers 5.x**（5.8.0+）。

### 1. 支持模型

#### 多模态向量模型

| 模型 | 模态 | 池化 | 推理 | 微调 | 描述 |
|:------|:----:|:----:|:----:|:----:|:-----|
| [Alibaba-NLP/gme-Qwen2-VL-2B-Instruct](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct) | 图像+文本 | last_token | Yes | Yes | 基于 Qwen2-VL 的 GME 多模态向量模型（2B） |
| [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 图像+文本 | last_token | Yes | Yes | Jina 多模态向量模型，内置多任务 LoRA 适配器 |
| [BAAI/BGE-VL-MLLM-S1](https://huggingface.co/BAAI/BGE-VL-MLLM-S1) | 图像+文本 | last_token | Yes | Yes | BGE-VL 系列多模态向量模型 |
| [BAAI/BGE-VL-MLLM-S2](https://huggingface.co/BAAI/BGE-VL-MLLM-S2) | 图像+文本 | last_token | Yes | Yes | BGE-VL 系列多模态向量模型 |
| [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | 图像+文本 | last_token | Yes | Yes | Qwen3-VL 多模态向量模型（2B） |
| [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | 图像+文本 | last_token | Yes | Yes | Qwen3-VL 多模态向量模型（8B） |

#### 多模态重排模型

| 模型 | 模态 | 推理 | 微调 | 损失 | 描述 |
|:------|:----:|:----:|:----:|:----:|:-----|
| [jinaai/jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0) | 图像+文本 | Yes | Yes | Pairwise / Listwise | 基于 Qwen2-VL 的多模态重排模型 |
| [Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | 图像+文本 | Yes | Yes | Pairwise / Listwise | Qwen3-VL 多模态重排模型（2B） |
| [Qwen/Qwen3-VL-Reranker-8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) | 图像+文本 | Yes | Yes | Pairwise / Listwise | Qwen3-VL 多模态重排模型（8B） |

#### 文本向量模型

| 模型 | 语言 | 池化 | 推理 | 微调 | 描述 |
|:------|:----:|:----:|:----:|:----:|:-----|
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Multilingual | last_token | Yes | Yes | Qwen3 文本向量模型（0.6B） |
| [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Multilingual | last_token | Yes | Yes | Qwen3 文本向量模型（4B） |
| [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | Multilingual | last_token | Yes | Yes | Qwen3 文本向量模型（8B） |

#### 文本重排模型

| 模型 | 语言 | 推理 | 微调 | 损失 | 描述 |
|:------|:----:|:----:|:----:|:----:|:-----|
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Multilingual | Yes | Yes | Pairwise / Listwise | Qwen3 文本重排模型（0.6B） |
| [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | Multilingual | Yes | Yes | Pairwise / Listwise | Qwen3 文本重排模型（4B） |
| [Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | Multilingual | Yes | Yes | Pairwise / Listwise | Qwen3 文本重排模型（8B） |

---

### 2. 推理

向量模型使用 `FlagAutoModel`，重排模型使用 `FlagAutoReranker`。所有模型均支持文本、图像和文本+图像输入。

#### 向量模型推理

加载多模态向量模型：

```python
from FlagEmbedding import FlagAutoModel

# GME-Qwen2-VL
model = FlagAutoModel.from_finetuned(
    'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct',
    model_class="gme-qwen2vl",
    use_fp16=True,
    devices="cuda:0",
    trust_remote_code=True,
)

# jina-embeddings-v4
model = FlagAutoModel.from_finetuned(
    'jinaai/jina-embeddings-v4',
    model_class="jina-embeddings-v4",
    use_fp16=True,
    devices="cuda:0",
    trust_remote_code=True,
)

# BGE-VL
model = FlagAutoModel.from_finetuned(
    'BAAI/BGE-VL-MLLM-S1',
    model_class="multimodal-mllm",
    use_fp16=True,
    devices="cuda:0",
    trust_remote_code=True,
)
```

编码查询和语料（文本、图像或两者混合）：

```python
# 纯文本
q_emb = model.encode_queries(queries=["What breed of dog is this?"])
p_emb = model.encode_corpus(corpus=["A golden retriever in a park."])

# 纯图像
q_emb = model.encode_queries(images=["query_image.jpg"])
p_emb = model.encode_corpus(images=["doc_image.jpg"])

# 文本 + 图像
q_emb = model.encode_queries(
    queries=["Describe the architecture"],
    images=["building.jpg"],
)

# 计算相似度
similarity = q_emb @ p_emb.T
```

#### 重排模型推理

加载多模态重排模型并计算分数：

```python
from FlagEmbedding import FlagAutoReranker

model = FlagAutoReranker.from_finetuned(
    'jinaai/jina-reranker-m0',
    use_fp16=True,
    devices="cuda:0",
)

# 文本-文本重排
scores = model.compute_score(
    [["What is AI?", "Artificial intelligence is ..."]],
    query_type="text",
    doc_type="text",
    normalize=True,
)

# 文本-图像重排
scores = model.compute_score(
    [["What breed of dog?", "dog_photo.jpg"]],
    query_type="text",
    doc_type="image",
    normalize=True,
)
```

---

### 3. 评估

#### MMEB-V2（视频任务）

内置 [MMEB-V2](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2)（Massive Multimodal Embedding Benchmark V2）视频任务评估模块，覆盖 **4 种任务类型**、**9 个数据集**：

| 任务类型 | 数据集 | 查询 | 语料 |
|:---------|:-------|:-----|:-----|
| 视频分类 | UCF101, HMDB51, Breakfast, K700, SSV2-actiontemplate | 视频网格图 + 指令 | 类别标签（文本） |
| 视频问答 | ActivityNetQA | 视频网格图 + 问题 | 答案文本 |
| 时刻检索 | Charades_STA, QVHighlight | 文本查询 | 视频片段网格图 |
| 视频检索 | SSV2 | 视频网格图 | 正/负例文本 |

视频帧均匀采样（默认 8 帧），缩放至 224x224，拼接为单张网格 PNG 图像，兼容基于图像的向量/重排 API。

##### 数据准备

1. 从 HuggingFace 下载 MMEB-V2 数据集：[TIGER-Lab/MMEB-V2](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2)
2. 在 `video-tasks/frames/` 下解压帧数据：
   ```bash
   cd /path/to/MMEB-V2/video-tasks/frames
   tar -xzf video_cls.tar.gz    # UCF101, HMDB51, Breakfast, K700, SSV2
   tar -xzf video_qa.tar.gz     # ActivityNetQA（可能分多个分卷）
   tar -xzf video_mret.tar.gz   # Charades_STA, QVHighlight
   tar -xzf video_ret.tar.gz    # SSV2
   ```
3. 确认 JSONL 数据文件在 `video-tasks/data/` 下（如 `ucf101.jsonl`、`activitynetqa.jsonl` 等）

##### 仅向量模型评估

```bash
python -m FlagEmbedding.evaluation.mmeb_v2 \
    --eval_name mmeb_v2 \
    --dataset_dir /path/to/MMEB-V2/video-tasks \
    --sub_datasets ucf101 hmdb51 breakfast k700 ssv2-actiontemplate activitynetqa charades_sta qvhighlight ssv2 \
    --max_frames 8 \
    --frame_size 224 \
    --splits test \
    --corpus_embd_save_dir ./mmeb_v2/corpus_embd \
    --output_dir ./mmeb_v2/search_results \
    --search_top_k 100 \
    --k_values 1 3 5 10 20 50 100 \
    --eval_output_method markdown \
    --eval_output_path ./mmeb_v2/eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_10 recall_at_50 map \
    --embedder_name_or_path BAAI/BGE-VL-MLLM-S1 \
    --embedder_model_class multimodal-mllm \
    --devices cuda:0 \
    --use_fp16
```

##### 两阶段评估（检索 + 重排）

添加 `--reranker_name_or_path` 和 `--rerank_top_k` 进行两阶段评估：

```bash
python -m FlagEmbedding.evaluation.mmeb_v2 \
    --eval_name mmeb_v2 \
    --dataset_dir /path/to/MMEB-V2/video-tasks \
    --sub_datasets ucf101 hmdb51 breakfast k700 ssv2-actiontemplate activitynetqa charades_sta qvhighlight ssv2 \
    --embedder_name_or_path BAAI/BGE-VL-MLLM-S1 \
    --embedder_model_class multimodal-mllm \
    --reranker_name_or_path jinaai/jina-reranker-m0 \
    --rerank_top_k 100 \
    --devices cuda:0 \
    --use_fp16
```

##### 快速测试模式

设置 `QUICK_TEST=true` 仅使用 50 个样本、3 个数据集快速验证：

```bash
QUICK_TEST=true bash examples/evaluation/mmeb_v2/eval_bge_vl.sh
```

##### 评估脚本

| 模型 | 类型 | 脚本 |
|:-----|:-----|:-----|
| BGE-VL (BAAI/BGE-VL-MLLM-S1) | 向量模型 | [`examples/evaluation/mmeb_v2/eval_bge_vl.sh`](examples/evaluation/mmeb_v2/eval_bge_vl.sh) |
| jina-embeddings-v4 | 向量模型 | [`examples/evaluation/mmeb_v2/eval_jina_v4.sh`](examples/evaluation/mmeb_v2/eval_jina_v4.sh) |
| Qwen3-VL-Embedding-2B | 向量模型 | [`examples/evaluation/mmeb_v2/eval_qwen3_vl_emb.sh`](examples/evaluation/mmeb_v2/eval_qwen3_vl_emb.sh) |
| jina-reranker-m0 | 重排模型 | [`examples/evaluation/mmeb_v2/eval_jina_m0.sh`](examples/evaluation/mmeb_v2/eval_jina_m0.sh) |
| Qwen3-VL-Reranker-2B | 重排模型 | [`examples/evaluation/mmeb_v2/eval_qwen3_vl_rer.sh`](examples/evaluation/mmeb_v2/eval_qwen3_vl_rer.sh) |

---

### 4. 微调

所有模型均支持通过 `torchrun` 进行 LoRA 微调。完整训练脚本位于 [`examples/finetune/`](examples/finetune/)。

#### 向量模型微调

```bash
torchrun --nproc_per_node 4 \
    -m FlagEmbedding.finetune.embedder.multimodal.base \
    --model_name_or_path Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --save_merged_lora_model True \
    --trust_remote_code True \
    --train_data path/to/train.jsonl \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --output_dir ./output \
    --learning_rate 1e-4 \
    --bf16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --deepspeed ds_stage1.json \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True
```

#### 重排模型微调

```bash
torchrun --nproc_per_node 2 \
    -m FlagEmbedding.finetune.reranker.multimodal.base \
    --model_name_or_path jinaai/jina-reranker-m0 \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --loss_type pairwise \
    --save_merged_lora_model True \
    --trust_remote_code True \
    --train_data path/to/train_with_scores.jsonl \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 1536 \
    --output_dir ./output \
    --learning_rate 5e-5 \
    --bf16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --deepspeed ds_stage0.json
```

#### 训练数据格式

**向量模型**（对比学习）— 每行一个 JSON 对象，包含 `query`、`pos` 和 `neg`：

```json
{
  "query": "What breed of dog is shown?",
  "pos": ["A Golden Retriever with a golden coat."],
  "neg": ["A Siamese cat with blue eyes.", "A tropical fish in an aquarium."]
}
```

**重排模型**（pairwise/listwise）— 额外包含 `pos_scores` 和 `neg_scores`：

```json
{
  "query": "What breed of dog is shown?",
  "pos": ["A Golden Retriever with a golden coat."],
  "pos_scores": [0.95],
  "neg": ["A Siamese cat with blue eyes.", "A tropical fish in an aquarium."],
  "neg_scores": [0.12, 0.05]
}
```

图像输入可通过 `query_image`、`pos_images` 和 `neg_images` 字段提供（文件路径或 `null`）。

#### MMEB-train 数据集（Parquet 格式）

除 JSON/JSONL 外，训练管线还支持 [TIGER-Lab/MMEB-train](https://huggingface.co/datasets/TIGER-Lab/MMEB-train) — 包含 20 个多模态任务子集的大规模训练数据集，采用 **Parquet** 格式。每个子集目录包含以下字段的 Parquet 文件：

| 字段 | 类型 | 描述 |
|:-----|:-----|:-----|
| `qry` | string | 查询文本（可能包含 `<\|image_1\|>` 占位符） |
| `qry_image_path` | string \| null | 查询图片相对路径 |
| `pos_text` | string | 正例文本 |
| `pos_image_path` | string \| null | 正例图片相对路径 |
| `neg_text` | string \| null | 负例文本 |
| `neg_image_path` | string \| null | 负例图片相对路径 |
| `instruction` | string | 任务指令 |

##### 数据准备

1. 从 HuggingFace 下载数据集：
   ```bash
   # 下载 MMEB-train（包含 Parquet 文件和图片 zip 压缩包）
   huggingface-cli download TIGER-Lab/MMEB-train --repo-type dataset --local-dir /path/to/MMEB-train
   ```

2. 解压图片：
   ```bash
   cd /path/to/MMEB-train
   python unzip_file.py  # 将 images_zip/*.zip 解压到 images/
   ```

##### 使用 `image_root_dir` 参数

由于 Parquet 文件中存储的是**相对**图片路径（如 `images/A-OKVQA/Train/xxx.jpg`），需通过 `--image_root_dir` 参数指定根目录来解析这些路径：

```bash
torchrun --nproc_per_node 4 \
    -m FlagEmbedding.finetune.embedder.multimodal.base \
    --model_name_or_path BAAI/BGE-VL-MLLM-S1 \
    --train_data /path/to/MMEB-train/A-OKVQA /path/to/MMEB-train/CIRR ... \
    --image_root_dir /path/to/MMEB-train/images \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --output_dir ./output \
    --bf16 \
    --num_train_epochs 3 \
    --deepspeed ds_stage1.json
```

每个 `--train_data` 条目是包含 Parquet 文件的**目录**。管线会自动检测 `.parquet` 文件并使用正确的字段映射加载。

##### 训练脚本

| 模型 | 类型 | 脚本 |
|:-----|:-----|:-----|
| BGE-VL (BAAI/BGE-VL-MLLM-S1) | 向量模型 | [`examples/finetune/embedder/multimodal/bge_vl_mmeb_train.sh`](examples/finetune/embedder/multimodal/bge_vl_mmeb_train.sh) |
| jina-embeddings-v4 | 向量模型 | [`examples/finetune/embedder/multimodal/jina_v4_mmeb_train.sh`](examples/finetune/embedder/multimodal/jina_v4_mmeb_train.sh) |
| Qwen3-VL-Embedding-2B | 向量模型 | [`examples/finetune/embedder/multimodal/qwen3_vl_emb_mmeb_train.sh`](examples/finetune/embedder/multimodal/qwen3_vl_emb_mmeb_train.sh) |
| jina-reranker-m0 | 重排模型 | [`examples/finetune/reranker/multimodal/jina_m0_mmeb_train.sh`](examples/finetune/reranker/multimodal/jina_m0_mmeb_train.sh) |
| Qwen3-VL-Reranker-2B | 重排模型 | [`examples/finetune/reranker/multimodal/qwen3_vl_rer_mmeb_train.sh`](examples/finetune/reranker/multimodal/qwen3_vl_rer_mmeb_train.sh) |

---

### 兼容性

- **transformers** >= 4.45（已测试 5.8.0）
- **torch** >= 2.0（已测试 2.10.0）
- **peft** >= 0.11
- 自动兼容 transformers 5.x 的 composite config、visual encoder 输出和 ROPE 初始化

---

<!-- ====== 以下为 FlagEmbedding 原始 README（中文） ====== -->

[<img src="./imgs/FlagOpen.png">](https://flagopen.baai.ac.cn/)

<h1 align="center">⚡️BGE: One-Stop Retrieval Toolkit For Search and RAG</h1>

![bge_logo](./imgs/bge_logo.jpg)

<p align="center">
    <a href="https://huggingface.co/collections/BAAI/bge-66797a74476eb1f085c7446d">
        <img alt="Build" src="https://img.shields.io/badge/BGE_series-🤗-yellow">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://huggingface.co/C-MTEB">
        <img alt="Build" src="https://img.shields.io/badge/C_MTEB-🤗-yellow">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/research/baai_general_embedding">
        <img alt="Build" src="https://img.shields.io/badge/FlagEmbedding-1.3.0-red">
    </a>
</p>
<h4 align="center">
    <p>
        <a href=#更新>更新</a> |
        <a href=#安装>安装</a> |
        <a href=#快速开始>快速开始</a> |
        <a href=#社区>社区</a> |
        <a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/research">项目</a> |
        <a href="#模型列表">模型列表</a> |
        <a href=#贡献者>贡献者</a> |
        <a href="#citation">Citation</a> |
        <a href="#license">License</a> 
    <p>
</h4>

[English](README.md) | [中文](README_zh.md)



## 更新

- 3/6/2025：:fire::fire: 推出 **BGE-VL**（[HF 仓库](https://huggingface.co/collections/BAAI/megapairs-67c6bbe49c15a9e7a7c69d92)），最先进的多模态向量模型，支持任意视觉搜索场景（文本到图像、图像到文本、图像+提示到图像、文本到图像+文本等）！采用 MIT 协议开源，学术和商业均可免费使用。同时发布 **MegaPairs**（[仓库](https://github.com/VectorSpaceLab/MegaPairs)，[论文](https://arxiv.org/abs/2412.14475)），大规模合成数据集。
- 12/5/2024：:book: 我们建立了 [BGE 文档网站](https://www.bge-model.com)，集中汇总 BGE 相关信息和资料！
- 10/29/2024：:earth_asia: 我们建立了 BGE 微信交流群，扫描 [二维码](./imgs/BGE_WeChat_Group.png) 加入群聊！
- <img src="./imgs/BGE_WeChat_Group.png" alt="bge_wechat_group" class="center" width="200">

- 10/22/2024：发布新模型 [OmniGen](https://github.com/VectorSpaceLab/OmniGen)，一个支持各种任务的统一图像生成模型。OmniGen 可以在不需要额外插件（如 ControlNet、IP-Adapter）或辅助模型的情况下完成复杂的图像生成任务。
- 9/10/2024：推出 **MemoRAG**，基于记忆启发的知识发现技术，是迈向 RAG 2.0 的关键一步（仓库：https://github.com/qhjqhj00/MemoRAG，论文：https://arxiv.org/pdf/2409.05591v1）
- 9/2/2024：开始维护更新[教程](./Tutorials/)，教程文件夹中的内容会在未来不断丰富，欢迎持续关注！:books:
- 7/26/2024：发布 [bge-en-icl](https://huggingface.co/BAAI/bge-en-icl)，一个结合了上下文学习能力的文本检索模型，通过提供与任务相关的查询-回答示例，可以编码语义更丰富的查询，进一步增强嵌入的语义表征能力。
- 7/26/2024：发布 [bge-multilingual-gemma2](https://huggingface.co/BAAI/bge-multilingual-gemma2)，基于 gemma-2-9b 的多语言文本向量模型，支持多种语言和多样的下游任务，在多语言检索数据集上取得最优结果。
- 7/26/2024：发布新的轻量级重排器 [bge-reranker-v2.5-gemma2-lightweight](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight)，基于 gemma-2-9b 的轻量级重排器，支持令牌压缩和分层轻量操作。:fire:

<details>
  <summary>更多</summary>

- 6/7/2024：发布首个专为长视频理解设计的全面评测基准 [MLVU](https://github.com/JUNJIE99/MLVU)。:fire:
- 5/21/2024：联合 Jina AI、Zilliz、HuggingFace 等机构发布评测基准 [AIR-Bench](https://github.com/AIR-Bench/AIR-Bench)，针对检索任务和 RAG 场景设计。[Leaderboard](https://huggingface.co/spaces/AIR-Bench/leaderboard) :fire:
- 4/30/2024：发布 [Llama-3-8B-Instruct-80K-QLoRA](https://huggingface.co/namespace-Pt/Llama-3-8B-Instruct-80K-QLoRA)，有效将 Llama-3-8B-Instruct 的上下文长度从 8K 扩展到 80K。[代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/longllm_qlora) :fire:
- 3/18/2024：发布新的 [rerankers](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/llm_reranker)，拥有更好的性能同时支持多语言和长文本。:fire:
- 3/18/2024：发布 [Visualized-BGE](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge)，赋予 BGE 视觉编码能力。:fire:
- 1/30/2024：发布 **BGE-M3**，第一个具有多功能、多语言和多粒度特性的文本检索模型。[技术报告](https://arxiv.org/pdf/2402.03216.pdf) 和 [代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3) :fire:
- 1/9/2024：发布 [Activation-Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon)，有效扩展大语言模型上下文长度。[技术报告](https://arxiv.org/abs/2401.03462)
- 12/24/2023：发布 **LLaRA**，基于 LLaMA-7B 的稠密检索模型。[技术报告](https://arxiv.org/abs/2312.15503) 和 [代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/LLARA)
- 11/23/2023：发布 [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/LM_Cocktail)，通过模型融合在微调时保持通用能力。[技术报告](https://arxiv.org/abs/2311.13534)
- 10/12/2023：发布 [LLM-Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/llm_embedder)，专为大语言模型各种检索增强任务设计。[技术报告](https://arxiv.org/pdf/2310.07554.pdf)
- 09/15/2023：发布 [技术报告](https://arxiv.org/pdf/2309.07597.pdf) 和 [数据集](https://data.baai.ac.cn/details/BAAI-MTP)
- 09/12/2023：新增重排模型和更新向量模型 bge-*-v1.5
- 09/07/2023：更新[微调代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/baai_general_embedding)
- 08/09/2023：BGE 模型整合入 Langchain；C-MTEB 中文榜单已[在线更新](https://huggingface.co/spaces/mteb/leaderboard)
- 08/05/2023：发布 base 和 small 规模模型
- 08/02/2023：:tada: :tada: 发布中英文向量模型 BGE，**在 MTEB 和 C-MTEB 榜单上取得最好的性能**
- 08/01/2023：发布大规模中文文本向量[评测榜单](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/C_MTEB)（**C-MTEB**），包括 31 个测试任务

</details>


BGE（BAAI General Embedding）专注于检索增强 LLM 领域，目前包括以下项目：

![projects](./imgs/projects.png)

- **推理**: [Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/embedder), [Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/reranker)
- **微调**: [Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder), [Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/reranker)
- **[评估](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/evaluation)**
- **[数据集](https://github.com/FlagOpen/FlagEmbedding/tree/master/dataset)**
- **[教程](https://github.com/FlagOpen/FlagEmbedding/tree/master/Tutorials)**
- **[研究](https://github.com/FlagOpen/FlagEmbedding/tree/master/research)**


## 安装
### 使用 pip:
如果你不需要微调模型，可以直接安装：
```
pip install -U FlagEmbedding
```
如果你需要微调模型，可以安装 finetune 依赖：
```
pip install -U FlagEmbedding[finetune]
```
### 从源码安装:

克隆并安装：
```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
# 不需要微调：
pip install  .
# 需要微调：
# pip install  .[finetune]
```
开发模式安装:
```
# 不需要微调：
pip install -e .
# 需要微调：
# pip install -e .[finetune]
```

## 快速开始
首先，加载一个 BGE 向量模型：
```
from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned('BAAI/bge-base-en-v1.5',
                                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                      use_fp16=True)
```
将语句作为模型输入，得到向量：
```
sentences_1 = ["I love NLP", "I love machine learning"]
sentences_2 = ["I love BGE", "I love text retrieval"]
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
```
取得向量后，通过内积计算相似度：
```
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```

更多细节请参考 [embedder 推理](./examples/inference/embedder)、[reranker 推理](./examples/inference/reranker)、[embedder 微调](./examples/finetune/embedder)、[reranker 微调](./examples/finetune/reranker)、[评估](./examples/evaluation)。

如果你对相关概念不熟悉，请查看[教程](./Tutorials/)。

## 社区

我们将持续维护 BGE 及 FlagEmbedding 社区，有任何想法建议都欢迎告诉我们！

近期会持续更新[教程](./Tutorials/)中的内容，希望为文本检索以及 RAG 打造出完整且详细的教学，欢迎持续关注！

在未来将会更新以下内容：

- 评估
- BGE-EN-ICL

<details>
  <summary>教程规划</summary>
    <img src="./Tutorials/tutorial_map.png"/>
</details>

## 模型列表

`bge` 是 `BAAI General Embedding` 的缩写。

| 模型 | 语言 | 描述 | 检索时的查询指令 |
|:------|:----:|:----:|:----:|
| [BAAI/bge-en-icl](https://huggingface.co/BAAI/bge-en-icl) | English | 基于 LLM 的向量模型，具有上下文学习能力 | 根据任务自由提供指示和少数示例 |
| [BAAI/bge-multilingual-gemma2](https://huggingface.co/BAAI/bge-multilingual-gemma2) | Multilingual | 基于 gemma-2-9b 的多语言向量模型 | 根据任务自由提供指示 |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Multilingual | 多功能（稠密、稀疏、多向量检索）、多语言、多粒度（8192 tokens） | |
| [LM-Cocktail](https://huggingface.co/Shitao) | English | 微调的 Llama 和 BGE 模型，可复现 LM-Cocktail 结果 | |
| [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder) | English | 专为 LLM 检索增强任务设计的向量模型 | 详见 [README](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/llm_embedder) |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Multilingual | 轻量级多语言交叉编码器，易于部署 | |
| [BAAI/bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma) | Multilingual | 多语言交叉编码器，英文和多语言能力表现出色 | |
| [BAAI/bge-reranker-v2-minicpm-layerwise](https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise) | Multilingual | 多语言交叉编码器，支持自由选择输出层加速推理 | |
| [BAAI/bge-reranker-v2.5-gemma2-lightweight](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight) | Multilingual | 多语言交叉编码器，支持选择输出层、压缩比例加速推理 | |
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | Chinese and English | 交叉编码器模型，精度比向量模型更高但推理效率较低 | |
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | Chinese and English | 交叉编码器模型，精度比向量模型更高但推理效率较低 | |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | English | 1.5 版本，相似度分布更合理 | `Represent this sentence for searching relevant passages: ` |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | English | 1.5 版本，相似度分布更合理 | `Represent this sentence for searching relevant passages: ` |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | English | 1.5 版本，相似度分布更合理 | `Represent this sentence for searching relevant passages: ` |
| [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) | Chinese | 1.5 版本，相似度分布更合理 | `为这个句子生成表示以用于检索相关文章：` |
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) | Chinese | 1.5 版本，相似度分布更合理 | `为这个句子生成表示以用于检索相关文章：` |
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | Chinese | 1.5 版本，相似度分布更合理 | `为这个句子生成表示以用于检索相关文章：` |
| [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) | English | 向量模型，将文本转换为向量 | `Represent this sentence for searching relevant passages: ` |
| [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en) | English | base-scale 向量模型 | `Represent this sentence for searching relevant passages: ` |
| [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) | English | small-scale 向量模型 | `Represent this sentence for searching relevant passages: ` |
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) | Chinese | 向量模型，将文本转换为向量 | `为这个句子生成表示以用于检索相关文章：` |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | Chinese | base-scale 向量模型 | `为这个句子生成表示以用于检索相关文章：` |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | Chinese | small-scale 向量模型 | `为这个句子生成表示以用于检索相关文章：` |



### 贡献者:
十分感谢所有参与 FlagEmbedding 社区成员的贡献，也欢迎新的成员加入！

<a href="https://github.com/FlagOpen/FlagEmbedding/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=FlagOpen/FlagEmbedding" />
</a>




## Citation

如果您觉得我们的工作有所帮助，请考虑点个星 :star: 和引用以下论文:

```
@misc{bge_m3,
  title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
  author={Chen, Jianlv and Xiao, Shitao and Zhang, Peitian and Luo, Kun and Lian, Defu and Liu, Zheng},
  year={2023},
  eprint={2309.07597},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@misc{cocktail,
      title={LM-Cocktail: Resilient Tuning of Language Models via Model Merging}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Xingrun Xing},
      year={2023},
      eprint={2311.13534},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{llm_embedder,
      title={Retrieve Anything To Augment Large Language Models}, 
      author={Peitian Zhang and Shitao Xiao and Zheng Liu and Zhicheng Dou and Jian-Yun Nie},
      year={2023},
      eprint={2310.07554},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}

@misc{bge_embedding,
      title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
      year={2023},
      eprint={2309.07597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
FlagEmbedding 基于 [MIT License](https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE) 开源协议。

## 原始版权
Copyright 2023 FlagOpen.  
源仓库: [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)  
Licensed under the Apache License, Version 2.0.
