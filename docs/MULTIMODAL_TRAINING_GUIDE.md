# 多模态模型训练适配指南

## 📋 完成的工作

本次适配为 BGE-VL-MLLM-S1 多模态模型创建了完整的训练框架，支持图像+文本的混合模态检索任务训练。

## 🗂️ 创建的文件列表

### 1. 核心训练代码 (`FlagEmbedding/finetune/embedder/multimodal/base/`)

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块导出，定义了所有公共类 |
| `__main__.py` | 训练入口点，解析参数并启动训练 |
| `arguments.py` | 多模态模型参数定义（继承自decoder_only） |
| `dataset.py` | **多模态数据集和Collator** |
| `modeling.py` | **多模态模型编码逻辑** |
| `load_model.py` | 模型加载和processor初始化 |
| `runner.py` | 训练运行器，协调整个训练流程 |
| `trainer.py` | 训练器，处理模型保存 |

### 2. 训练脚本 (`examples/finetune/embedder/multimodal/`)

| 文件 | 说明 |
|------|------|
| `base.sh` | 基础训练脚本 |
| `base_same_dataset.sh` | 知识蒸馏+同数据集批次训练 |
| `README.md` | 详细使用文档 |

## 🔑 关键技术实现

### 1. **多模态数据处理** (`dataset.py`)

数据集支持的字段格式：
```json
{
  "query": "文本查询",
  "qry_image_path": "查询图像路径",
  "pos_text": ["正例文本"],
  "pos_image_path": ["正例图像路径"],
  "neg_text": ["负例文本"],
  "neg_image_path": ["负例图像路径"],
  "pos_scores": [分数],  // 知识蒸馏
  "neg_scores": [分数]
}
```

**关键特性**：
- 自动处理文本、图像、或文本+图像的组合
- 支持None值（纯文本或纯图像）
- 使用 `model.data_process()` 替代tokenizer

### 2. **多模态编码** (`modeling.py`)

```python
def encode(self, features):
    # 使用模型的output_hidden_states获取输出
    outputs = self.model(**features, output_hidden_states=True)
    
    # 取最后一个token作为embedding
    embeddings = outputs[:, -1, :]
    
    if self.normalize_embeddings:
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    
    return embeddings.contiguous()
```

**与纯文本模型的区别**：
- ✅ 不使用 `last_hidden_state` + `last_token_pool`
- ✅ 使用 `output_hidden_states=True` + 直接取 `[:, -1, :]`
- ✅ features已由 `model.data_process()` 处理，包含多模态信息

### 3. **Processor初始化** (`load_model.py`, `runner.py`)

```python
# 加载模型后初始化processor
if hasattr(model, 'set_processor'):
    model.set_processor(model_name_or_path)

# Collator使用模型的data_process方法
self.data_collator = MultimodalEmbedderCollator(
    model=self.model.model,  # 传入base model
    ...
)
```

### 4. **Collator实现** (`dataset.py`)

```python
def __call__(self, features):
    # 处理queries
    queries_inputs = self.model.data_process(
        text=query_texts,
        images=query_images,
        q_or_c='q',
        task_instruction=None
    )
    
    # 处理passages
    passages_inputs = self.model.data_process(
        text=passages_texts,
        images=passages_images,
        q_or_c='c',
        task_instruction=None
    )
    
    return {
        "queries": queries_inputs,
        "passages": passages_inputs,
        "teacher_scores": teacher_scores,
        "no_in_batch_neg_flag": False
    }
```

## 🚀 使用方法

### 1. 准备数据

确保你的数据集（如 TIGER-Lab/MMEB-train）包含以下字段：
- `query` / `qry` - 查询文本
- `qry_image_path` - 查询图像路径
- `pos_text` / `pos_image_path` - 正例
- `neg_text` / `neg_image_path` - 负例

### 2. 修改训练脚本

编辑 `examples/finetune/embedder/multimodal/base.sh`：

```bash
# 更新训练数据路径
train_data="TIGER-Lab/MMEB-train"

# 设置GPU数量
num_gpus=4

# 调整batch size和epochs
num_train_epochs=3
per_device_train_batch_size=4
```

### 3. 启动训练

```bash
cd examples/finetune/embedder/multimodal
bash base.sh
```

### 4. 使用知识蒸馏（可选）

如果数据包含teacher scores：

```bash
bash base_same_dataset.sh
```

## 📊 训练参数说明

### 模型参数
```bash
--model_name_or_path BAAI/BGE-VL-MLLM-S1  # 基础模型
--use_lora True                            # 使用LoRA
--lora_rank 32                             # LoRA秩
--lora_alpha 64                            # LoRA alpha
--trust_remote_code True                   # 必须设置
--save_merged_lora_model True              # 保存合并模型
```

### 数据参数
```bash
--train_data <path>              # 训练数据
--train_group_size 8            # 每个query的passage数
--query_max_len 512             # query最大长度
--passage_max_len 512           # passage最大长度
--knowledge_distillation False  # 是否使用KD
```

### 训练参数
```bash
--learning_rate 1e-4            # 学习率
--temperature 0.02              # 对比学习温度
--normalize_embeddings True     # 归一化
--negatives_cross_device        # 跨设备负例
--gradient_checkpointing        # 梯度检查点（节省内存）
```

## 🔄 训练流程

```
1. 加载模型 (load_model.py)
   └─> 初始化processor (model.set_processor)
   
2. 加载数据 (dataset.py)
   └─> MultimodalEmbedderTrainDataset
   
3. 创建Collator (dataset.py)
   └─> 使用 model.data_process() 处理多模态输入
   
4. 创建模型 (modeling.py)
   └─> BiMultimodalEmbedderModel
   
5. 训练 (trainer.py + runner.py)
   └─> 计算对比损失/KD损失
   
6. 保存模型
   └─> 可选：合并LoRA权重
```

## ⚙️ 与纯文本训练的区别

| 方面 | 纯文本 (decoder_only) | 多模态 (multimodal) |
|------|---------------------|-------------------|
| **数据输入** | 只有文本 | 文本+图像路径 |
| **预处理** | `tokenizer()` | `model.data_process()` |
| **编码方式** | `last_hidden_state` + `last_token_pool` | `output_hidden_states` + `[:, -1, :]` |
| **Collator** | `DataCollatorWithPadding` | `MultimodalEmbedderCollator` |
| **Processor** | 不需要 | 需要 `model.set_processor()` |

## 🐛 故障排查

### 1. ImportError: trust_remote_code
**解决**：确保设置 `--trust_remote_code True`

### 2. AttributeError: 'Model' object has no attribute 'data_process'
**解决**：确认模型是 BGE-VL-MLLM-S1 且正确加载了processor

### 3. OOM Error
**解决**：
- 减小 `per_device_train_batch_size`
- 启用 `gradient_checkpointing`
- 使用DeepSpeed ZeRO优化

### 4. 数据格式错误
**解决**：检查数据是否包含正确的字段名（qry_image_path等）

## 📁 输出文件

训练后会生成：
```
output_multimodal_bge_vl_mllm_s1/
├── checkpoint-500/
├── checkpoint-1000/
├── merged_model/          # 如果启用save_merged_lora_model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
├── trainer_state.json
└── training_args.bin
```

## 🎯 下一步

训练完成后，使用推理代码：

```python
from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned(
    './output_multimodal_bge_vl_mllm_s1/merged_model',
    devices="cuda:0"
)

embeddings = model.encode(
    sentences="查询文本",
    images="path/to/image.jpg",
    q_or_c="q"
)
```

## ✅ 总结

本次适配实现了：
1. ✅ 完整的多模态训练框架
2. ✅ 支持文本、图像、文本+图像的混合输入
3. ✅ 兼容知识蒸馏
4. ✅ 支持LoRA高效训练
5. ✅ 多GPU分布式训练
6. ✅ DeepSpeed优化

所有代码都基于decoder_only版本扩展，保持了架构一致性，同时完全支持多模态特性！🎉

