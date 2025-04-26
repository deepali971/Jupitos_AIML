import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Extract embeddings from BERT model
class TransformerEncoderWithFNN(nn.Module):
    def __init__(self, model):
        super(TransformerEncoderWithFNN, self).__init__()
        self.bert = model
        self.ffn = FeedForwardNetwork(768, 512, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        output = self.ffn(cls_embedding)
        return output

# Instantiate the model
model = TransformerEncoderWithFNN(model)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

# Sample input text
text = ["This is a great movie!", "This is a terrible movie."]
labels = torch.tensor([1, 0], dtype=torch.float32)

# Tokenize input text
encodings = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# Training loop
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask).squeeze()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
