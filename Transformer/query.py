# Query
# This module implements a simple query mechanism for attention mechanisms.

def create_query(embedding):
    """
    Creates a query vector from the input embedding.
    
    Args:
        embedding (list): The input embedding vector.
        
    Returns:
        list: A query vector derived from the embedding.
    """
    # For simplicity, we will just return the embedding as the query
    return embedding

# Example usage
if __name__ == "__main__":
    sample_embedding = [0.1, 0.2, 0.3, 0.4]
    print(create_query(sample_embedding))
