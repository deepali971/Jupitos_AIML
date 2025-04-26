# Tokenization
# This module implements a simple tokenization process for text input.

def tokenize(text):
    """
    Tokenizes the input text into words.
    
    Args:
        text (str): The input text to be tokenized.
        
    Returns:
        list: A list of tokens (words).
    """
    # Split the text into words based on whitespace
    tokens = text.split()
    return tokens

# Example usage
if __name__ == "__main__":
    sample_text = "Hello, this is a sample text for tokenization."
    print(tokenize(sample_text))
