from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import BertTokenizer, BertModel

app = Flask(__name__)


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Load the model
model_path = 'improved_model.pth'
input_size = 768  # Ensure this matches your input data size
num_classes = 36  # Ensure this matches the number of classes
model = SimpleNN(input_size=input_size, num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased', cache_dir='./bert')

df = pd.read_csv('final.csv')
unique_tags = df['tag'].unique()
tag_to_int = {tag: i for i, tag in enumerate(unique_tags)}
int_to_tag = {i: tag for tag, i in tag_to_int.items()}


def encode_text(texts, max_length=128):
    inputs = tokenizer(texts, return_tensors='pt',
                       max_length=max_length, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = bert(**inputs)
    # Use the mean of the last hidden state as the text representation
    return outputs.last_hidden_state.mean(dim=1)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['input']

    # Encode the input text
    input_tensor = encode_text([input_text])

    # Perform prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_tag = int_to_tag[predicted.item()]

    response = {
        'prediction': predicted_tag
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
