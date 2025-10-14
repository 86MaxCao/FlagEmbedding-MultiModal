# 多模态模型适配说明 (Multimodal Model Adaptation)

## 概述

本次修改实现了对多模态MLLM（Multi-modal Large Language Model）模型 BGE-VL-MLLM-S1 的完整适配，使其能够在FlagEmbedding-MultiModal项目框架下运行。

## 修改文件列表

### 1. 核心库文件

#### 1.1 `FlagEmbedding/inference/embedder/multimodal/base.py`
**主要修改：**
- 在 `__init__` 方法中添加了对 `set_processor` 的调用，用于初始化多模态模型的处理器
- 完全重写了 `encode_single_device` 方法：
  - 添加了新参数：`images`（图像路径）、`q_or_c`（query/candidate标识）、`task_instruction`（任务指令）
  - 添加了对多模态模型的检测逻辑（检查 `data_process` 方法是否存在）
  - 对于多模态模型：使用 `model.data_process()` 处理输入，并从输出中取最后一个token的hidden state作为embedding
  - 保留了对纯文本模型的支持（原有逻辑）

**代码片段：**
```python
# 在 __init__ 中添加
if hasattr(self.model, 'set_processor'):
    self.model.set_processor(model_name_or_path)

# encode_single_device 支持多模态
if hasattr(self.model, 'data_process'):
    # 多模态处理逻辑
    inputs = self.model.data_process(
        images=images,
        text=sentences,
        q_or_c=q_or_c,
        task_instruction=task_instruction
    )
    outputs = self.model(**inputs, output_hidden_states=True)
    embeddings = outputs[:, -1, :]  # 取最后一个token
```

#### 1.2 `FlagEmbedding/inference/embedder/model_mapping.py`
**主要修改：**
- 在 `EmbedderModelClass` 枚举中添加了 `MULTIMODAL_MLLM` 类型
- 在 `EMBEDDER_CLASS_MAPPING` 中添加了多模态模型类的映射
- 在 `BGE_MAPPING` 中添加了 `BGE-VL-MLLM-S1` 模型的配置

**代码片段：**
```python
class EmbedderModelClass(Enum):
    # ... existing ...
    MULTIMODAL_MLLM = "multimodal-mllm"

EMBEDDER_CLASS_MAPPING = OrderedDict([
    # ... existing ...
    (EmbedderModelClass.MULTIMODAL_MLLM, FlagMLLMModel)
])

BGE_MAPPING = OrderedDict([
    # ... existing ...
    (
        "BGE-VL-MLLM-S1",
        EmbedderConfig(FlagMLLMModel, PoolingMethod.LAST_TOKEN, trust_remote_code=True)
    ),
])
```

#### 1.3 `FlagEmbedding/inference/auto_embedder.py`
**说明：** 无需修改，该文件已经可以正确处理多模态模型，因为它只是根据 `model_mapping.py` 中的配置来选择相应的类。

### 2. 示例文件

#### 2.1 `examples/inference/embedder/multimodal/auto_mllm_single_device.py`
使用 `FlagAutoModel` 自动加载多模态模型的示例，包含三个测试用例：
- 多模态（图像+文本）查询
- 纯文本查询
- 纯图像查询

#### 2.2 `examples/inference/embedder/multimodal/base_mllm_single_device.py`
直接使用 `FlagMLLMModel` 基类加载多模态模型的示例。

#### 2.3 `examples/inference/embedder/multimodal/README.md`
多模态模型使用文档，包含：
- 使用说明
- 参数说明
- 示例代码
- 故障排查

## 使用方法

### 方法1：使用 FlagAutoModel（推荐）

```python
from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned(
    'BAAI/BGE-VL-MLLM-S1',
    devices="cuda:0",
)

# 多模态查询
query_emb = model.encode(
    sentences="Make the background dark",
    images="./path/to/image.png",
    q_or_c="q",
    task_instruction="Retrieve the target image: "
)

# 候选图像
candidate_emb = model.encode(
    images=["./candi1.png", "./candi2.png"],
    q_or_c="c"
)

scores = query_emb @ candidate_emb.T
```

### 方法2：直接使用 FlagMLLMModel

```python
from FlagEmbedding import FlagMLLMModel

model = FlagMLLMModel(
    model_name_or_path='BAAI/BGE-VL-MLLM-S1',
    normalize_embeddings=True,
    use_fp16=True,
    devices="cuda:0",
    trust_remote_code=True,
)

# 使用方式同上
```

## 关键技术点

### 1. 处理器初始化
多模态模型需要特殊的处理器来处理图像和文本输入。通过检测 `set_processor` 方法的存在来自动初始化处理器。

### 2. 多模态输入处理
- 使用模型的 `data_process` 方法来处理图像和文本
- 支持三种模式：纯文本、纯图像、图像+文本

### 3. Embedding提取
- 多模态模型：从 `output_hidden_states=True` 的输出中提取最后一个token的hidden state
- 文本模型：使用 `last_token_pool` 函数提取

### 4. 兼容性设计
- 通过 `hasattr` 检查模型是否支持多模态功能
- 保留了对纯文本模型的完整支持
- 向后兼容原有的API

## 测试

运行测试脚本：

```bash
# 测试 FlagAutoModel
cd examples/inference/embedder/multimodal
python auto_mllm_single_device.py

# 测试 FlagMLLMModel
python base_mllm_single_device.py
```

## 注意事项

1. **trust_remote_code**: 多模态模型需要设置 `trust_remote_code=True`
2. **图像路径**: 确保图像路径正确且格式支持（PNG, JPG等）
3. **内存**: 多模态模型较大，建议使用GPU并注意显存使用
4. **批处理**: 目前多模态模型的批处理是通过 `data_process` 方法自动处理的

## 与原始实现的对比

### 原始实现 (inference_bge_vl_mllm_s1.py)
```python
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.set_processor(MODEL_NAME)
inputs = model.data_process(text=..., images=..., q_or_c=..., task_instruction=...)
outputs = model(**inputs, output_hidden_states=True)[:, -1, :]
```

### 适配后的实现
```python
model = FlagAutoModel.from_finetuned('BAAI/BGE-VL-MLLM-S1')
embeddings = model.encode(sentences=..., images=..., q_or_c=..., task_instruction=...)
```

适配后的实现保持了相同的底层逻辑，但提供了更统一和易用的API接口。

