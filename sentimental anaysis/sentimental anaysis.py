#  sentiment analysis 
import torch
from transformers import AutoTokenizer, BertModel, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader

# Step 1: Define the Model Architecture
class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(SentimentAnalysisModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token
        logits = self.classifier(pooled_output)
        return logits

# Step 2: Load and Preprocess the Dataset
def load_and_preprocess_data():
    dataset = load_dataset("sentiment140")  # Load Sentiment140 dataset (tweets)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Convert datasets to PyTorch tensors
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized_train, tokenized_test, tokenizer

# Step 3: Prepare Data Loaders
def create_data_loaders(tokenized_train, tokenized_test, batch_size=16):
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_test, batch_size=batch_size)
    return train_loader, test_loader

# Step 4: Train the Model
def train_model(model, train_loader, optimizer, loss_fn, epochs=3):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} completed")

# Step 5: Evaluate the Model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

# Step 6: Save the Model
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

# Step 7: Predict Sentiment on New Text
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    labels = ["negative", "positive"]
    return labels[predicted_class]

# Main Function
def main():
    # Load and preprocess data
    tokenized_train, tokenized_test, tokenizer = load_and_preprocess_data()

    # Create data loaders
    train_loader, test_loader = create_data_loaders(tokenized_train, tokenized_test)

    # Initialize the model
    model = SentimentAnalysisModel("bert-base-uncased", num_labels=2)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    print("Training the model...")
    train_model(model, train_loader, optimizer, loss_fn, epochs=3)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, test_loader)

    # Save the model
    save_model(model, "one_for_all_sentiment_model.pth")

    # Example prediction
    text = "I absolutely loved this movie! It was fantastic."
    print(f"Predicted sentiment: {predict_sentiment(text, model, tokenizer)}")

if __name__ == "__main__":
    main()