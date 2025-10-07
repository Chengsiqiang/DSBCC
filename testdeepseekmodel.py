from huggingface_hub import InferenceClient
import os
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score
from collections import defaultdict



#deepseek-R1-distill endpoint
#https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
deepseek_model_client = InferenceClient(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",

)

def make_prediction(model, system_role, user_query):

    response = model.chat_completion(
    messages=[{"role": "system", "content": system_role},
        {"role": "user", "content": user_query}],
    max_tokens=1000,
    )

    return response.choices[0].message.content

system_role = "Assign positive, negative, or neutral sentiment to the movie review. Return only a single word in your response"
user_query = "I like this movie a lot"
output = make_prediction(deepseek_model_client,
               system_role,
               user_query)
output