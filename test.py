import pickle
import numpy as np
import os
import math

data_dir = "./data"
vocab_file = os.path.join(data_dir, "vocab.zh.pkl")
with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f, encoding='bytes')
word_emb = np.load('nnlm_word_embeddings.zh.npy')
#vocab = {v : k for k, v in vocab.items()}
word1_id = vocab["中国"]
word2_id = vocab["美国"]

word1_emb = word_emb[word1_id]

word2_emb = word_emb[word2_id]

def cosin_distance(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)

print(cosin_distance(word1_emb,word2_emb))