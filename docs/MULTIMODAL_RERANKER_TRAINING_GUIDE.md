# 多模态Reranker训练适配指南

## 📋 完成的工作

成功为 `jinaai/jina-reranker-m0` 多模态reranker创建了完整的训练框架，支持 **Pairwise Loss** 和 **Listwise Loss** 两种损失函数。

## 🗂️ 创建的文件列表

### 1. 核心训练代码 (7个文件)

| 文件 | 说明 |
|------|------|
| `FlagEmbedding/finetune/reranker/multimodal/__init__.py` | 模块初始化 |
| `FlagEmbedding/finetune/reranker/multimodal/base/__init__.py` | 导出类 |
| `FlagEmbedding/finetune/reranker/multimodal/base/__main__.py` | 训练入口 |
| `FlagEmbedding/finetune/reranker/multimodal/base/arguments.py` | 参数定义 |
| `FlagEmbedding/finetune/reranker/multimodal/base/modeling.py` | ⭐ 模型+Loss实现 |
| `FlagEmbedding/finetune/reranker/multimodal/base/dataset.py` | ⭐ 多模态数据集 |
| `FlagEmbedding/finetune/reranker/multimodal/base/load_model.py` | 模型加载 |
| `FlagEmbedding/finetune/reranker/multimodal/base/runner.py` | ⭐ 训练运行器 |
| `FlagEmbedding/finetune/reranker/multimodal/base/trainer.py` | 训练器 |

### 2. 训练脚本 (4个文件)

| 文件 | 说明 |
|------|------|
| `examples/finetune/reranker/multimodal/base_pairwise.sh` | Pairwise Loss训练 |
| `examples/finetune/reranker/multimodal/base_listwise.sh` | Listwise Loss训练 |
| `examples/finetune/reranker/multimodal/README.md` | 使用文档 |
| `examples/finetune/reranker/multimodal/example_data_format.jsonl` | 数据格式示例 |

### 3. 更新的文件 (3个文件)

| 文件 | 修改内容 |
|------|---------|
| `FlagEmbedding/inference/reranker/model_mapping.py` | 添加MULTIMODAL_BASE类型 |
| `FlagEmbedding/inference/reranker/__init__.py` | 导出MultimodalReranker |
| `FlagEmbedding/inference/__init__.py` | 顶层导出 |

## 🎯 两种Loss实现

### 1. Pairwise Loss（默认推荐）

**原理**: 使用Margin Ranking Loss，对每个正例-负例对进行比较。

```python
def compute_pairwise_loss(self, scores):
    # 分离正例和负例分数
    pos_scores = scores[:, 0]  # 第一个总是正例
    neg_scores = scores[:, 1:]  # 其余都是负例
    
    # 对每个负例计算: pos_score > neg_score
    loss = MarginRankingLoss(pos_scores, neg_scores, target=1)
    
    return loss
```

**优点**:
- ✅ 训练更稳定
- ✅ 适合小数据集
- ✅ 关注相对排序
- ✅ 不需要teacher scores

**适用场景**:
- 大多数重排序任务
- 没有teacher scores
- 数据质量参差不齐

### 2. Listwise Loss

**原理**: 使用Cross-Entropy Loss，将所有候选一起考虑。

```python
def compute_listwise_loss(self, scores):
    # 所有候选的分数
    grouped_logits = scores.view(batch_size, -1)
    
    # 目标：第一个位置（正例）
    target = torch.zeros(batch_size, dtype=torch.long)
    
    # Cross-entropy loss
    loss = CrossEntropyLoss(grouped_logits, target)
    
    return loss
```

**优点**:
- ✅ 全局最优
- ✅ 分数更calibrated
- ✅ 配合KD效果好

**适用场景**:
- 有可靠的teacher scores
- 需要校准的分数
- 大规模数据集

## 🔑 核心实现细节

### 1. **多模态数据处理** (dataset.py)

```python
def __getitem__(self, item):
    data = self.dataset[item]
    
    # 提取query（文本和/或图像）
    query_text = data.get('query', data.get('qry', None))
    query_image = data.get('query_image', data.get('qry_image_path', None))
    
    # 提取passages（文本和/或图像）
    pos_texts = data.get('pos', data.get('pos_text', []))
    pos_images = data.get('pos_images', data.get('pos_image_path', []))
    neg_texts = data.get('neg', data.get('neg_text', []))
    neg_images = data.get('neg_images', data.get('neg_image_path', []))
    
    # 采样并返回
    return {
        'query_text': query_text,
        'query_image': query_image,
        'passages_text': passages_text,
        'passages_image': passages_image,
        'teacher_scores': teacher_scores
    }
```

### 2. **Collator处理** (dataset.py)

```python
def __call__(self, features):
    # 格式化prompts
    for q, d in zip(queries, passages):
        prompt = formatting_prompts_func(q, d, query_type, doc_type)
    
    # 加载图像
    images = load_images_batch(image_paths)
    
    # 使用processor处理
    batch = self.processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    
    # 添加score token
    batch["input_ids"] = torch.cat([
        batch["input_ids"],
        torch.full((batch_size, 1), score_token_id)
    ], dim=1)
    
    return {"pair": batch, "teacher_scores": scores}
```

### 3. **Loss计算** (modeling.py)

```python
class MultimodalRerankerModel(AbsRerankerModel):
    def __init__(self, ..., loss_type='pairwise'):
        self.loss_type = loss_type
        self.pairwise_loss = nn.MarginRankingLoss(margin=0.0)
        self.listwise_loss = nn.CrossEntropyLoss()
    
    def forward(self, pair, teacher_scores):
        scores = self.encode(pair)  # Get model scores
        
        if self.training:
            if self.loss_type == 'pairwise':
                loss = self.compute_pairwise_loss(scores, teacher_scores)
            else:  # listwise
                loss = self.compute_listwise_loss(scores, teacher_scores)
        
        return RerankerOutput(loss=loss, scores=scores)
```

## 📊 数据格式

### 标准格式

```json
{
  "query": "查询文本（可选）",
  "qry_image_path": "查询图像路径（可选）",
  "pos": ["正例文本"],
  "pos_image_path": ["正例图像路径"],
  "neg": ["负例文本1", "负例文本2"],
  "neg_image_path": ["负例图像1", "负例图像2"],
  "pos_scores": [0.95],
  "neg_scores": [0.3, 0.2]
}
```

### 支持的字段别名

- `query` = `qry`
- `query_image` = `qry_image_path`
- `pos` = `pos_text`
- `pos_images` = `pos_image_path`
- `neg` = `neg_text`
- `neg_images` = `neg_image_path`

### 模态组合示例

**1. 纯文本**:
```json
{"query": "text", "pos": ["pos text"], "neg": ["neg text"]}
```

**2. 文本→图像**:
```json
{"query": "text", "pos_image_path": ["pos.png"], "neg_image_path": ["neg.png"]}
```

**3. 图像→文本**:
```json
{"qry_image_path": "query.png", "pos": ["pos text"], "neg": ["neg text"]}
```

**4. 图像→图像**:
```json
{"qry_image_path": "query.png", "pos_image_path": ["pos.png"], "neg_image_path": ["neg.png"]}
```

**5. 混合**:
```json
{"query": "text", "qry_image_path": "query.png", "pos": ["text"], "pos_image_path": ["image.png"]}
```

## 🚀 使用方法

### 训练

```bash
cd examples/finetune/reranker/multimodal

# 使用Pairwise Loss（默认推荐）
bash base_pairwise.sh

# 使用Listwise Loss（配合KD）
bash base_listwise.sh
```

### 推理

```python
from FlagEmbedding import FlagAutoReranker

# 加载训练好的模型
reranker = FlagAutoReranker.from_finetuned(
    './output_multimodal_reranker_jina_m0_pairwise/merged_model',
    devices="cuda:0"
)

# 重排序
query = "machine learning"
docs = ["doc1.png", "doc2.png", "doc3.png"]
pairs = [[query, doc] for doc in docs]

scores = reranker.compute_score(
    pairs,
    query_type='text',
    doc_type='image'
)

# 按分数排序
ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
```

## 🔄 训练流程

```
1. 加载模型 (load_model.py)
   └─> AutoModel + AutoProcessor
   └─> 应用LoRA（如果启用）

2. 加载数据 (dataset.py)
   └─> MultimodalRerankerTrainDataset
   └─> 支持文本+图像字段

3. 创建Collator (dataset.py)
   └─> MultimodalRerankerCollator
   └─> 使用processor处理多模态输入
   └─> 添加score token

4. 创建模型 (modeling.py)
   └─> MultimodalRerankerModel
   └─> 选择loss类型（pairwise/listwise）

5. 训练 (trainer.py + runner.py)
   └─> 计算loss
   └─> 反向传播

6. 保存模型
   └─> 可选：合并LoRA权重
```

## 📐 Loss对比

| 特性 | Pairwise Loss | Listwise Loss |
|------|--------------|---------------|
| **公式** | MarginRankingLoss | CrossEntropyLoss |
| **比较方式** | 1 pos vs 每个 neg | 所有候选一起 |
| **稳定性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **收敛速度** | 中等 | 较快 |
| **KD支持** | ✅ | ✅ |
| **数据要求** | 低 | 中等 |
| **分数校准** | 一般 | 好 |
| **推荐场景** | 通用 | 有teacher scores |

## 🎓 使用建议

### 选择Pairwise Loss当:
1. ✅ 没有teacher scores
2. ✅ 数据集较小
3. ✅ 想要稳定训练
4. ✅ 关注排序而非绝对分数

### 选择Listwise Loss当:
1. ✅ 有高质量teacher scores
2. ✅ 数据集较大
3. ✅ 需要校准的分数值
4. ✅ 配合知识蒸馏

## 🔧 参数配置

### Pairwise Loss配置

```bash
--loss_type pairwise
--knowledge_distillation False  # 可选
--learning_rate 5e-5
--train_group_size 8            # 1 pos + 7 neg
```

### Listwise Loss配置

```bash
--loss_type listwise
--knowledge_distillation True   # 推荐开启
--learning_rate 5e-5
--train_group_size 8
```

## 🧪 训练示例

### 最小示例

```bash
# 1. 准备数据（参考example_data_format.jsonl）
cat > train.jsonl << 'EOF'
{"query": "ML paper", "pos_image_path": ["paper.png"], "neg_image_path": ["cat.png"]}
EOF

# 2. 修改训练脚本
vim base_pairwise.sh
# 改: train_data="train.jsonl"

# 3. 启动训练
bash base_pairwise.sh
```

### 完整流程

```bash
# 1. 进入目录
cd examples/finetune/reranker/multimodal

# 2. 准备数据
# 确保数据包含query, pos, neg字段（文本和/或图像）

# 3. 选择loss类型并训练
# 方式1: Pairwise Loss
bash base_pairwise.sh

# 方式2: Listwise Loss (需要teacher scores)
bash base_listwise.sh

# 4. 等待训练完成，模型保存在output_dir

# 5. 使用训练好的模型
python test_trained_model.py
```

## 💡 技术亮点

### 1. **灵活的Loss机制**

```python
# modeling.py
if self.loss_type == 'pairwise':
    # 每个正例与每个负例比较
    loss = pairwise_loss(pos_scores, neg_scores)
elif self.loss_type == 'listwise':
    # 所有候选一起比较
    loss = cross_entropy(all_scores, target=0)
```

### 2. **多模态输入支持**

```python
# dataset.py
# 自动检测并处理多种模态组合
query_type = 'image' if has_query_image else 'text'
doc_type = 'image' if has_doc_image else 'text'

# 格式化prompt
prompt = formatting_prompts_func(query, doc, query_type, doc_type)
```

### 3. **知识蒸馏集成**

```python
# 两种loss都支持KD
if teacher_scores is not None:
    kd_loss = -torch.mean(
        torch.sum(
            torch.log_softmax(logits, dim=-1) * teacher_targets, 
            dim=-1
        )
    )
    loss += kd_loss
```

### 4. **Processor集成**

```python
# load_model.py
processor = AutoProcessor.from_pretrained(
    model_name_or_path,
    max_pixels=602112,
    min_pixels=3136,
    trust_remote_code=True
)
model._processor = processor
```

## 📈 Loss公式详解

### Pairwise Loss

$$L_{pairwise} = \frac{1}{N \times M} \sum_{i=1}^{N} \sum_{j=1}^{M} \max(0, margin - (s_{pos}^i - s_{neg}^{ij}))$$

其中：
- \(N\) = batch size
- \(M\) = 负例数量
- \(s_{pos}^i\) = 第i个正例的分数
- \(s_{neg}^{ij}\) = 第i个query的第j个负例的分数

### Listwise Loss

$$L_{listwise} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(s_{pos}^i)}{\sum_{j=0}^{M} \exp(s_j^i)}$$

等价于 CrossEntropy，目标总是位置0（正例）。

### 知识蒸馏Loss

$$L_{KD} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=0}^{M} p_j^{teacher} \log p_j^{student}$$

其中 \(p_j\) 是softmax归一化后的概率。

## 🎯 总结

| 方面 | Pairwise | Listwise |
|------|----------|----------|
| **默认推荐** | ✅ | - |
| **稳定性** | 高 | 中等 |
| **需要KD** | 否 | 推荐是 |
| **分数校准** | 一般 | 好 |
| **训练速度** | 中等 | 快 |
| **内存占用** | 低 | 中等 |

### 最佳实践

1. **首选Pairwise Loss**: 适合90%的场景
2. **有teacher scores时**: 尝试Listwise Loss + KD
3. **数据质量差时**: 使用Pairwise Loss
4. **需要精确分数时**: 使用Listwise Loss

所有代码已准备就绪，可以开始训练多模态reranker了！🎉

