import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Key
# This module implements a simple key mechanism for attention mechanisms.

def create_key(embedding):
    """
    Creates a key vector from the input embedding.
    
    Args:
        embedding (list): The input embedding vector.
        
    Returns:
        list: A key vector derived from the embedding.
    """
    # For simplicity, we will just return the embedding as the key
    return embedding

# Example usage
if __name__ == "__main__":
    text = "Transformers are powerful NLP models!"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(text)
    
    embedding_layer = nn.Embedding(30522, 768)  # Embedding layer maps token IDs to dense vectors
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    embeddings = embedding_layer(input_ids)
    
    key_vector = create_key(embeddings)
    print("Key Vector Shape:", key_vector.shape)
