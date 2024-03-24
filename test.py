from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'  # You can replace this with any other BERT variant
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Sample document and query
document_text = "Đây là một văn bản trong tiếng Việt."
query_text = "Đây là câu hỏi trong tiếng Việt."

document_tokens = tokenizer.encode(document_text, add_special_tokens=True)
query_tokens = tokenizer.encode(query_text, add_special_tokens=True)

document_tensor = torch.tensor([document_tokens])
query_tensor = torch.tensor([query_tokens])

model.eval()

with torch.no_grad():
    document_embeddings = model(document_tensor)[0][:, 0, :]  # Take the embedding of [CLS] token
    query_embeddings = model(query_tensor)[0][:, 0, :]        # Take the embedding of [CLS] token

# Print embeddings
print("Document Embedding Shape:", document_embeddings.shape)
print("Query Embedding Shape:", query_embeddings.shape)
cosine_sim = cosine_similarity(document_embeddings, query_embeddings)

print("Cosine Similarity between Document and Query:", cosine_sim.item())

