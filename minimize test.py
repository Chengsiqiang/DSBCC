import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer, Qwen2ForSequenceClassification

# 模型路径（替换为你自己的路径）
MODEL_PATH = "/home/user/Desktop/csq/model"

# 加载分词器和模型
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Qwen2ForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=104,
        torch_dtype=torch.float16,
        ignore_mismatched_sizes=True,
        pad_token_id=tokenizer.pad_token_id,
        #gradient_checkpointing=True  # 显存优化
    ).to("cuda")
    model.resize_token_embeddings(len(tokenizer))  # 调整嵌入层大小
    print("✅ 模型加载成功！")
except Exception as e:
    print("❌ 模型加载失败:", str(e))
    raise
#for name, param in model.named_parameters():
    #print(f"{name}: {param.device}")  # 所有参数应显示 'cuda:0'


# 测试输入（模拟代码片段）
test_code = """
int main() {
    int a = 1;
    int b = 2;
    return a + b;
}
"""

# 分词处理
try:
    inputs = tokenizer(test_code, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}  # 显式移动到 GPU
    print("✅ 分词成功！输入形状:", {k: v.shape for k, v in inputs.items()})
except Exception as e:
    print("❌ 分词失败:", str(e))
    raise

# 前向传播测试
try:
    with torch.no_grad():
        outputs = model(**inputs)
    print("✅ 前向传播成功！输出 logits 形状:", outputs.logits.shape)
    print("示例输出 logits:", outputs.logits[0].cpu().numpy()[:5])  # 打印前5个类别分数
except Exception as e:
    print("❌ 前向传播失败:", str(e))
    raise