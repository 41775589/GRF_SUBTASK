import together, os
from together import Together
import json
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_embeddings(input_texts, model_api_string, TOGETHER_API_KEY):
    """Generate embeddings from Together python library.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
    """
    together_client = together.Together(api_key = TOGETHER_API_KEY)
    outputs = together_client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    )
    return np.array([x.embedding for x in outputs.data])


def generate_batch_embeddings(input_texts, model_api_string, batch_size):
    """将大的文本列表分批生成嵌入"""
    all_embeddings = []

    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}...")  # 打印当前批次
        batch_embeddings = generate_embeddings(batch, model_api_string)
        all_embeddings.append(batch_embeddings)

    # 将所有批次拼接起来
    return np.concatenate(all_embeddings, axis=0)