# FlagEmbedding-MultiModal  
> FlagEmbedding 多模态扩展（非官方）

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-≥3.8-green.svg)

## 声明
本仓库是由 [@86MaxCao](https://github.com/86MaxCao) 维护的**个人 fork**，
与 [FlagOpen](https://github.com/FlagOpen) **无关联**。  
所有原始代码遵循 Apache 2.0 协议；详见 [LICENSE](LICENSE)。



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

- 5/11/2026：完全兼容 **transformers 5.8.0 + torch 2.10.0**（`latest` 环境）。所有兼容性补丁统一收录于 `FlagEmbedding/compat.py`。新增 **Qwen3-VL-Reranker 微调**支持（2B/8B）。新增 **Qwen3-Reranker** 纯文本推理与微调（0.6B/4B/8B）。新增 **Qwen3-Embedding** 微调（0.6B/4B/8B）。jina-reranker-m0 微调在 latest 环境下可用。详见 [CHANGELOG](docs/CHANGELOG_20260511_164720.md)。
- 5/10/2025：新增 **Qwen3-VL-Embedding**（2B/8B）多模态微调支持（InfoNCE 对比损失），以及 **jina-reranker-m0** 的 pairwise/listwise loss 微调。新增 **Qwen3-VL-Embedding**、**Qwen3-VL-Reranker** 和 **Qwen3-Embedding** 模型的推理支持。详见 [CHANGELOG](docs/CHANGELOG.md)。
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

### 多模态向量模型

| 模型 | 模态 | 池化 | 推理 | 微调 | 描述 |
|:------|:----:|:----:|:----:|:----:|:-----|
| [BAAI/BGE-VL-MLLM-S1](https://huggingface.co/BAAI/BGE-VL-MLLM-S1) | 图像+文本 | last_token | Yes | Yes | BGE-VL 系列多模态向量模型 |
| [BAAI/BGE-VL-MLLM-S2](https://huggingface.co/BAAI/BGE-VL-MLLM-S2) | 图像+文本 | last_token | Yes | Yes | BGE-VL 系列多模态向量模型 |
| [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | 图像+文本 | last_token | Yes | Yes | Qwen3-VL 多模态向量模型（2B） |
| [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | 图像+文本 | last_token | Yes | Yes | Qwen3-VL 多模态向量模型（8B） |

### 多模态重排模型

| 模型 | 模态 | 推理 | 微调 | 损失 | 描述 |
|:------|:----:|:----:|:----:|:----:|:-----|
| [jinaai/jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0) | 图像+文本 | Yes | Yes | Pairwise / Listwise | 基于 Qwen2-VL 的多模态重排模型 |
| [Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | 图像+文本 | Yes | Yes | Pairwise / Listwise | Qwen3-VL 多模态重排模型（2B） |
| [Qwen/Qwen3-VL-Reranker-8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) | 图像+文本 | Yes | Yes | Pairwise / Listwise | Qwen3-VL 多模态重排模型（8B） |

### 文本重排模型（新增支持）

| 模型 | 语言 | 推理 | 微调 | 损失 | 描述 |
|:------|:----:|:----:|:----:|:----:|:-----|
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Multilingual | Yes | Yes | Pairwise / Listwise | Qwen3 文本重排模型（0.6B） |
| [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | Multilingual | Yes | Yes | Pairwise / Listwise | Qwen3 文本重排模型（4B） |
| [Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | Multilingual | Yes | Yes | Pairwise / Listwise | Qwen3 文本重排模型（8B） |

### 文本向量模型（新增支持）

| 模型 | 语言 | 池化 | 推理 | 微调 | 描述 |
|:------|:----:|:----:|:----:|:----:|:-----|
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Multilingual | last_token | Yes | Yes | Qwen3 文本向量模型（0.6B） |
| [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Multilingual | last_token | Yes | Yes | Qwen3 文本向量模型（4B） |
| [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | Multilingual | last_token | Yes | Yes | Qwen3 文本向量模型（8B） |



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
