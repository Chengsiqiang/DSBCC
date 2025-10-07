import os
import warnings
import torch
import pandas as pd
from datasets import Dataset, ClassLabel, Features, Value
from transformers import (
    AutoTokenizer,
    Qwen2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score,matthews_corrcoef
import numpy as np

import logging
# 1: 在最开头设置警告过滤
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)

# 2: 设置日志级别
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# 3: 禁用特定模块的日志
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 禁用TensorFlow日志（如果有）
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 禁用CUDA阻塞警告



#   关键修复点1: 确保tokenizer并行安全
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置参数
MODEL_PATH = "/home/user/Desktop/csq/model"
MAX_LENGTH = 1024
BATCH_SIZE = 16
GRAD_ACCUM = 2


# 数据预处理
def preprocess_data(df):
    df['code'] = df['code'].fillna('').astype(str)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    invalid_mask = df['label'].isna() | ~df['label'].between(1, 104)
    if invalid_mask.any():
        print(f"发现 {invalid_mask.sum()} 个无效样本:")
        df = df[~invalid_mask].copy()
    df['label'] = (df['label'] - 1).astype(int)
    return df


# 加载数据集
train_df = preprocess_data(pd.read_csv("/home/user/Desktop/csq/divide-data/train_set.csv", encoding='utf-8'))
val_df = preprocess_data(pd.read_csv("/home/user/Desktop/csq/divide-data/eval_set.csv", encoding='utf-8'))
test_df = preprocess_data(pd.read_csv("/home/user/Desktop/csq/divide-data/test_set.csv", encoding='utf-8'))

#   关键修复点2: 彻底解决填充token问题
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

#   关键修复点3: 强制重新设置填充token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("添加新的[PAD] token")
else:
    #   关键修复点4: 确保填充token有效
    try:
        pad_token_id = tokenizer.pad_token_id
        print(f"原始填充token: '{tokenizer.pad_token}' (ID: {pad_token_id})")

        # 测试填充功能
        test_pad = tokenizer.pad_token_id
        if test_pad is None or tokenizer.decode([test_pad]).strip() == "":
            raise ValueError("无效的填充token")
    except Exception:
        print("原始填充token无效，添加新的[PAD] token")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#  关键修复点5: 确保填充token正确设置
if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print(f"最终填充token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")


#  🔧 关键修复点6: 安全的数据集创建函数
def create_safe_dataset(df):
    input_ids = []
    attention_mask = []
    labels = []

    for _, row in df.iterrows():
        # 🔧 关键修复点7: 使用encode_plus确保正确处理
        encoding = tokenizer.encode_plus(
            row['code'],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # 不在此填充
            return_attention_mask=True,
            return_tensors=None
        )
        input_ids.append(encoding["input_ids"])
        attention_mask.append(encoding["attention_mask"])
        labels.append(row['label'])

    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels
    })


# 创建数据集
train_dataset = create_safe_dataset(train_df)
val_dataset = create_safe_dataset(val_df)
test_dataset = create_safe_dataset(test_df)

# 打印样本验证
print("数据集样本验证:")
sample = train_dataset[0]
print(f"输入ID类型: {type(sample['input_ids'][0])}")
print(f"输入ID长度: {len(sample['input_ids'])}")
print(f"注意力掩码类型: {type(sample['attention_mask'][0])}")
print(f"标签值: {sample['label']}")


#  🔧 关键修复点8: 自定义数据整理器确保正确填充
class SafeDataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["label"] for item in batch]

        #  🔧 关键修复点9: 手动填充并截断
        padded_inputs = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            max_length=self.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            return_attention_mask=True
        )

        return {
            "input_ids": padded_inputs["input_ids"],
            "attention_mask": padded_inputs["attention_mask"],
            "labels": torch.tensor(labels)
        }


data_collator = SafeDataCollator(tokenizer, MAX_LENGTH)

# 验证数据整理器
print("验证数据整理器...")
sample_batch = [train_dataset[0], train_dataset[1]]
collated = data_collator(sample_batch)
print(f"整理后批次形状: input_ids={collated['input_ids'].shape}, attention_mask={collated['attention_mask'].shape}")

# 加载模型
model = Qwen2ForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=104,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

#  🔧 关键修复点10: 模型加载后调整
# 如果添加了新token，调整模型嵌入层
if tokenizer.pad_token == "[PAD]" and tokenizer.pad_token_id != 151643:
    model.resize_token_embeddings(len(tokenizer))
    print(f"模型嵌入层已调整为 {len(tokenizer)} 个token")

# 启用梯度检查点
model.gradient_checkpointing_enable()
model.config.use_cache = False

# 初始化分类头
if hasattr(model, "classifier"):
    torch.nn.init.xavier_uniform_(model.classifier.weight)
    print("分类头权重已初始化")
elif hasattr(model, "score"):
    torch.nn.init.xavier_uniform_(model.score.weight)
    print("分类头权重已初始化")

# 训练参数
training_args = TrainingArguments(
    output_dir="./deepseek-finetuned",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=3,
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to="none",
    fsdp="",
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=True
)


# 评估函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    macro_recall = recall_score(labels, preds, average="macro")
    macro_precision = precision_score(labels, preds, average="macro")
    mcc = matthews_corrcoef(labels, preds)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "mcc": mcc
    }


# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

#  🔧 关键修复点11: 改进的预热检查
print("=== 改进的预热检查 ===")
try:
    # 创建单个样本批次
    sample = train_dataset[0]
    sample_batch = [sample, sample]  # 两个相同样本

    # 使用整理器
    batch = data_collator(sample_batch)

    # 移动到设备
    inputs = {
        "input_ids": batch["input_ids"].to(model.device),
        "attention_mask": batch["attention_mask"].to(model.device),
        "labels": batch["labels"].to(model.device)
    }

    # 前向传播
    model.train()
    outputs = model(**inputs)

    # 反向传播
    loss = outputs.loss
    loss.backward()
    print(f"预热检查成功! Loss: {loss.item():.4f}")

    # 显存使用情况
    for i in range(torch.cuda.device_count()):
        print(f"GPU{i} 最大显存使用: {torch.cuda.max_memory_allocated(i) / 1024 ** 3:.1f}GB")

except Exception as e:
    print(f"预热检查失败: {str(e)}")
    #  🔧 关键修复点12: 彻底解决填充问题
    print("强制添加新填充token并调整模型")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"新填充token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # 重新测试
    try:
        batch = data_collator(sample_batch)
        inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
            "labels": batch["labels"].to(model.device)
        }
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        print(f"第二次预热检查成功! Loss: {loss.item():.4f}")
    except Exception as e2:
        print(f"第二次预热检查失败: {str(e2)}")
        # 如果仍然失败，禁用梯度检查点
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        training_args.gradient_checkpointing = False
        print("已禁用梯度检查点")



# 开始训练
try:
    print("=== 开始训练 ===")
    print(f"训练参数: batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, max_length={MAX_LENGTH}")

    # 使用小数据集测试
    small_train = train_dataset.select(range(100))
    trainer.train_dataset = small_train
    print("使用前100个样本进行测试训练...")
    trainer.train()

    # 完整训练
    print("开始完整训练...")
    trainer.train_dataset = train_dataset
    trainer.train()

    # 最终评估
    test_results = trainer.evaluate(test_dataset)
    print("\n=== 最终测试集评估结果 ===")
    print(f"测试集准确率: {test_results['eval_accuracy']:.4f}")
    print(f"宏平均F1: {test_results['eval_macro_f1']:.4f}")
    print(f"宏平均召回率: {test_results['eval_macro_recall']:.4f}")
    print(f"宏平均精确率: {test_results['eval_macro_precision']:.4f}")
    print(f"马修斯相关系数(MCC): {test_results['eval_mcc']:.4f}")
    print(f"验证损失: {test_results['eval_loss']:.4f}")

    # 保存模型
    trainer.save_model("./deepseek-finetuned-final")
    tokenizer.save_pretrained("./deepseek-finetuned-final")
    print("模型保存成功!")

except RuntimeError as e:
    print(f"训练错误: {str(e)}")
    if "CUDA out of memory" in str(e):
        print("显存不足! 请尝试:")
        print(f"1. 降低 MAX_LENGTH (当前: {MAX_LENGTH})")
        print(f"2. 降低 BATCH_SIZE (当前: {BATCH_SIZE})")
        print(f"3. 增加 GRAD_ACCUM (当前: {GRAD_ACCUM})")
    elif "shape" in str(e):
        print("维度不匹配! 请检查:")
        print("- 模型输出维度是否与标签数量一致")
        print("- tokenizer 分词结果是否包含 input_ids 和 attention_mask")
except KeyboardInterrupt:
    print("训练被手动中断")
    trainer.save_model("./deepseek-finetuned-interrupted")
    print("已保存中断时的模型状态")
except Exception as e:
    print(f"未知错误: {str(e)}")







