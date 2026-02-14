# SigLIP2 CoreML 转换与 ImageNet Zero-Shot 测评笔记

## 项目目标

将 HuggingFace 的 `google/siglip2-base-patch16-224` 模型转换为 CoreML 格式（Vision + Text 分离的 `.mlpackage`），并在 ImageNet-1K 验证集上进行 zero-shot 分类准确率测评，对比三种后端：

1. HuggingFace (PyTorch) 原始模型
2. CoreML FP32
3. CoreML INT8 量化

---

## 最终测评结果

数据集：ImageNet-1K 验证集 1%（500 张图片），seed=42

| 后端 | Top-1 | Top-5 |
|---|---|---|
| HuggingFace (PyTorch) | 71.40% | 91.00% |
| CoreML FP32 | 71.60% | 90.80% |
| CoreML INT8 | 67.20% | 88.40% |

**结论**：CoreML FP32 与 HF 原始模型精度一致（差异在采样噪声范围内）。INT8 量化损失约 4 个百分点 Top-1，属于 INT8 的正常范围。

---

## 文件清单

| 文件 | 说明 |
|---|---|
| `convert_siglip2_to_coreml.py` | CoreML 转换脚本（Vision + Text FP32 导出 + INT8 量化 + 自动验证） |
| `evaluate_siglip2_imagenet.py` | ImageNet zero-shot 测评脚本（支持 HF / CoreML FP32 / CoreML INT8） |
| `SigLIP2_Vision.mlpackage/` | CoreML Vision 模型（FP32） |
| `SigLIP2_Text.mlpackage/` | CoreML Text 模型（FP32） |
| `SigLIP2_Vision_INT8.mlpackage/` | CoreML Vision 模型（INT8） |
| `SigLIP2_Text_INT8.mlpackage/` | CoreML Text 模型（INT8） |
| `google/siglip2-base-patch16-224/` | HF 原始模型文件（config.json, model.safetensors, tokenizer 等） |
| `datasets/data/` | ImageNet-1K 验证集（14 个 parquet 分片，共 50000 张） |
| `datasets/classes.py` | `IMAGENET2012_CLASSES` OrderedDict（1000 个 synset→label 映射） |

---

## 遇到的关键问题与解决方案

### 问题 1：CoreML Text 模型内部生成 attention_mask 导致精度崩溃

**现象**：CoreML FP32/INT8 的 Top-1 准确率仅 ~3%，而 HF 原始模型达到 ~71%。

**根因**：`TextModelWrapper` 在 `forward()` 中自行计算了 attention_mask：

```python
# 错误写法（旧版）
attention_mask = (input_ids != self.pad_token_id).long()
outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
```

SigLIP2 的文本模型在 `attention_mask=None` 时会内部使用全 1 的 mask。该模型在训练时就是这样处理 padding 的——padding token 不会被 mask 掉。一旦显式传入 mask（把 padding 位置置 0），embedding 就会完全偏离，zero-shot 分类直接失效。

**修复**：删除 mask 生成，不传 `attention_mask`：

```python
# 正确写法（当前版）
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    input_ids = input_ids.long()
    outputs = self.text_model(input_ids=input_ids)  # attention_mask 默认 None → 全 1
    return outputs.pooler_output
```

**验证数据**：
- Vision embedding CoreML vs HF cosine similarity: 0.9999（无问题）
- Text embedding 旧版（带 mask）vs HF（无 mask）cosine similarity: ~0.586（严重偏离）
- Text embedding 修复后 CoreML vs HF cosine similarity: 0.9999（正常）

### 问题 2：文本必须全部小写

**现象**：使用原始大小写的类名做 zero-shot prompt，准确率明显偏低。

**根因**：SigLIP2 是基于 Gemma tokenizer 的，模型训练时使用了小写文本。当前 `transformers` 版本附带的是 `GemmaTokenizerFast`，不会自动小写（自动小写是新版 `Siglip2Tokenizer` / `Siglip2Processor` 的功能，但 pip 安装的 transformers 版本还没有）。

**修复**：手动对整个 prompt 调用 `.lower()`：

```python
prompts = [PROMPT_TEMPLATE.format(name.lower()).lower() for name in class_names]
```

### 问题 3：pad_token_id 为 0 时的 Python 假值陷阱

**现象**：`pad_token_id = processor.tokenizer.pad_token_id` 返回 `0`。如果写 `pad_token_id or 1`，由于 `0` 在 Python 中是 falsy，会错误地变成 `1`。

**修复**：使用 `if pad_token_id is None` 做判断，而不是依赖 truthy/falsy。转换脚本中已封装为 `resolve_pad_token_id()` 函数。（注意：修复 attention_mask 问题后，测评脚本不再需要 pad_token_id，但转换脚本的 `resolve_pad_token_id` 仍保留用于其他可能的用途。）

### 问题 4：Zero-shot prompt 模板必须正确

**要求**：必须使用 `"This is a photo of {label}."` 格式（含句号），这是 SigLIP2 文档中明确规定的 pipeline 默认模板。

来源：https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md

> "To get the same results as the Pipeline, a prompt template of 'This is a photo of {label}.' should be passed."

### 问题 5：不要使用 pipeline API

HF `pipeline("zero-shot-image-classification")` API 在处理大规模评估时效率低且行为不透明。正确做法是直接用 `AutoModel`，分别调用 `model.get_image_features()` 和 `model.get_text_features()` 获取 embedding，再通过 cosine similarity 做排序。

---

## 转换脚本关键设计

### Vision 模型 (`VisionModelWrapper`)

- 输入：`ct.ImageType`，RGB 图片，`(1, 3, 224, 224)`
- 内部做 normalize：`(image * rescale_factor - mean) / std`
- 输出：`pooler_output` embedding `(1, 768)`

### Text 模型 (`TextModelWrapper`)

- 输入：`input_ids` Int32 `(1, 64)`
- **不做** attention_mask 计算（模型内部默认全 1 mask）
- 输出：`pooler_output` embedding `(1, 768)`
- seq_len=64（来自 `config.text_config.max_position_embeddings`）

### INT8 量化

使用 `coremltools.optimize.coreml.linear_quantize_weights`，`linear_symmetric` 模式，`weight_threshold=0`（量化所有权重）。

### 验证阈值

| 指标 | FP32 阈值 | INT8 阈值 |
|---|---|---|
| max_abs_diff | ≤ 0.03 | ≤ 0.7 |
| cosine_similarity | ≥ 0.999 | ≥ 0.99 |

---

## 测评脚本关键设计

### 数据加载

- ImageNet-1K 验证集存储为 14 个 parquet 分片（`validation-*.parquet`）
- 每行包含 `image`（dict with `bytes`）和 `label`（int 0-999）
- 通过 `--sample-ratio` 控制采样比例，默认 0.01（500 张）

### Zero-shot 分类流程

1. 对 1000 个 ImageNet 类名生成 prompt：`"this is a photo of {label}."`（全小写）
2. 用 tokenizer 编码为 `input_ids`，padding 到 `max_length=64`
3. 计算所有 1000 个类的 text embedding，L2 归一化
4. 对每张图片计算 image embedding，L2 归一化
5. 计算 `image_embeds @ text_embeds.T` 得到 `(N, 1000)` 的相似度矩阵
6. 取 Top-1 / Top-5 计算准确率

### HF 推理注意事项

- `model.get_text_features(input_ids=ids)` —— **不传 attention_mask**
- `model.get_image_features(pixel_values=pixel_values)` —— 正常传入

---

## 环境依赖

```
torch >= 2.7
transformers >= 4.49
coremltools >= 8.x
pyarrow
Pillow
numpy
```

注意：`coremltools` 对 torch 版本有兼容性警告（2.7.1 未经官方测试，2.7.0 是最新测试版本），实际使用中未遇到问题。
