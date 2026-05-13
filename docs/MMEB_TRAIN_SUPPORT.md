# MMEB-train 数据集支持

本文档描述了为 FlagEmbedding-MultiModal 添加的 MMEB-train 数据集评估支持。

## 概述

MMEB-train 是一个多模态评估基准训练数据集，包含多种任务类型：
- **文本问答**：图像+问题 → 文本答案
- **图像检索**：图像+文本查询 → 目标图像
- **混合任务**：支持多种模态组合

## 数据集结构

```
MMEB-train-flat/
├── DocVQA/
│   ├── diverse_instruction-00000-of-00001.parquet
│   ├── original-00000-of-00001.parquet
│   └── train-00000-of-00001.parquet
├── CIRR/
│   └── *.parquet
├── MSCOCO/
│   └── *.parquet
├── MSCOCO_i2t/
│   └── *.parquet
├── MSCOCO_t2i/
│   └── *.parquet
├── VisualNews_i2t/
│   └── *.parquet
├── VisualNews_t2i/
│   └── *.parquet
├── Visual7W/
│   └── *.parquet
├── WebQA/
│   └── *.parquet
└── images/
    ├── DocVQA/Train/DocVQA_image_0.jpg
    ├── CIRR/train/image1.jpg
    └── ...
```

## 数据格式

每个 parquet 文件包含以下字段：

```python
{
    'qry': 'what is the date mentioned in this letter?',  # 查询问题
    'qry_image_path': 'images/DocVQA/Train/DocVQA_image_0.jpg',  # 查询图像路径
    'pos_text': '1/8/93',  # 正确答案文本
    'pos_image_path': '',  # 正确答案图像路径（可能为空）
    'neg_text': '398,627.92',  # 错误答案文本
    'neg_image_path': ''  # 错误答案图像路径（可能为空）
}
```

## 新增文件

### 评估模块
- `FlagEmbedding/evaluation/mmeb_train/`
  - `__init__.py` - 模块初始化
  - `arguments.py` - 评估参数定义
  - `data_loader.py` - 数据加载器
  - `searcher.py` - 检索器和重排序器
  - `evaluator.py` - 评估器
  - `runner.py` - 运行器
  - `__main__.py` - 主入口

### 评估脚本
- `examples/evaluation/mmeb_train/`
  - `eval_mmeb_train_embedder.sh` - 嵌入器评估脚本
  - `eval_mmeb_train_reranker.sh` - 重排序器评估脚本

## 主要特性

### 1. 多子数据集支持
支持评估多个子数据集：
- DocVQA
- CIRR  
- MSCOCO
- MSCOCO_i2t
- MSCOCO_t2i
- VisualNews_i2t
- VisualNews_t2i
- Visual7W
- WebQA

### 2. 混合模态支持
支持多种任务类型：
- **text**: 纯文本问答任务
- **image**: 图像检索任务
- **mixed**: 混合模态任务

### 3. 快速测试模式
支持快速测试功能：
- `max_samples`: 限制评估样本数量
- `skip_corpus_cache`: 跳过语料库缓存
- `QUICK_TEST=true`: 快速测试模式

### 4. 多模态检索和重排序
- 支持 `BAAI/BGE-VL-MLLM-S1` 嵌入器
- 支持 `jinaai/jina-reranker-m0` 重排序器
- 支持多种模态组合的检索和重排序

## 使用方法

### 快速测试（前100个样本）
```bash
# 在脚本中取消注释这一行
QUICK_TEST=true

# 然后运行
bash examples/evaluation/mmeb_train/eval_mmeb_train_embedder.sh
bash examples/evaluation/mmeb_train/eval_mmeb_train_reranker.sh
```

### 完整评估
```bash
# 保持 QUICK_TEST 注释状态
bash examples/evaluation/mmeb_train/eval_mmeb_train_embedder.sh
bash examples/evaluation/mmeb_train/eval_mmeb_train_reranker.sh
```

### 自定义配置
```bash
python -m FlagEmbedding.evaluation.mmeb_train \
    --eval_name mmeb_train \
    --dataset_dir /path/to/MMEB-train-flat \
    --sub_datasets DocVQA CIRR MSCOCO \
    --task_types text image mixed \
    --max_samples 1000 \
    --skip_corpus_cache True \
    --embedder_name_or_path BAAI/BGE-VL-MLLM-S1 \
    --reranker_name_or_path jinaai/jina-reranker-m0
```

## 评估指标

支持的评估指标：
- `ndcg_at_10`: NDCG@10
- `recall_at_10`: Recall@10  
- `recall_at_50`: Recall@50
- `map`: Mean Average Precision

## 技术实现

### 数据加载
- 从本地 parquet 文件加载数据
- 支持多子数据集遍历
- 处理相对路径图像文件
- 支持样本数量限制

### 检索和重排序
- 支持文本+图像的查询
- 支持文本和图像答案的检索
- 使用 FAISS 进行高效相似度搜索
- 支持多模态重排序

### 缓存机制
- 支持语料库嵌入缓存
- 支持快速测试模式（跳过缓存）
- 支持增量评估

## 与 MegaPairs 的区别

| 特性 | MegaPairs | MMEB-train |
|------|-----------|------------|
| 任务类型 | 图像检索 | 混合模态（文本问答+图像检索） |
| 目标类型 | 图像 | 文本+图像 |
| 子数据集 | 单一数据集 | 多个子数据集 |
| 数据格式 | 固定格式 | 灵活的任务类型配置 |
| 评估复杂度 | 中等 | 高（支持多种模态组合） |

## 注意事项

1. **路径配置**: 确保 `dataset_dir` 指向正确的 MMEB-train-flat 目录
2. **图像路径**: 图像路径是相对路径，需要与根目录拼接
3. **内存使用**: 大型数据集可能需要较多内存，建议使用快速测试模式先验证
4. **模型兼容性**: 确保使用的模型支持多模态输入输出

## 故障排除

### 常见问题
1. **路径错误**: 检查 `dataset_dir` 和图像路径是否正确
2. **内存不足**: 减少 `max_samples` 或使用 `skip_corpus_cache=True`
3. **模型加载失败**: 检查模型名称和路径是否正确
4. **数据格式错误**: 确保 parquet 文件格式正确
5. **导入错误**: 确保安装了必要的依赖包（torch, transformers, datasets, tqdm, faiss, numpy）

### 调试建议
- 使用快速测试模式验证配置
- 检查日志输出中的错误信息
- 确认数据加载是否正常
- 验证模型推理是否成功
- 检查 Python 环境和依赖包安装

### 已修复的问题
1. **AttributeError: 'MMEBTrainEvalArgs' object has no attribute 'add_args'**
   - 修复：使用 `HfArgumentParser` 替代 `add_args` 方法
   - 文件：`__main__.py`

2. **导入错误**
   - 修复：移除未使用的导入，优化导入结构
   - 文件：`data_loader.py`, `searcher.py`, `runner.py`
