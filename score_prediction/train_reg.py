import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import psycopg2
from urllib.parse import urlparse
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import re
from collections import Counter
from datasets import load_dataset
import json

# -----------------------------
# 1. LOAD 80% TRAINING & VALIDATION DATA
# -----------------------------
print("Fetching Hacker News data...")
conn = psycopg2.connect("postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")
cur = conn.cursor()
cur.execute('''SELECT i.title, i.url, i.score, u.id, u.karma 
FROM hacker_news.items AS i
JOIN hacker_news.users AS u ON i.by = u.id  -- Use INNER JOIN to exclude null users
WHERE title IS NOT NULL AND score IS NOT NULL AND u.karma IS NOT NULL AND (abs(hashtext(i.id::text)) % 100) >= 20
''')
rows = cur.fetchall()
conn.close()

titles, urls, scores, by, karmas = zip(*rows)

# -----------------------------
# 2. Preprocess titles
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    return text.split()

tokenized_titles = [preprocess(title) for title in titles]
word_counts = Counter(word for title in tokenized_titles for word in title)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}
vocab_size = len(word_to_ix)


# -----------------------------
# 3. Load Vocabulary
# -----------------------------
with open("vocab-w2-200titles.json", "r", encoding="utf-8") as f:
    word_to_ix = json.load(f)

ix_to_word = {int(i): w for w, i in word_to_ix.items()}
vocab_size = len(word_to_ix)

# -----------------------------
# 3. Load Pre-trained Embeddings
# -----------------------------
embed_dim = 300  
embeddings = torch.load("embeddings-w2-200titles-300dim-10e.pt", map_location='cpu')  # Shape: [vocab_size, embed_dim]

assert embeddings.shape[0] == vocab_size, "Vocab size mismatch!"


# -----------------------------
# 4. CBOW Model
# -----------------------------
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).mean(dim=1)
        return self.linear(embeds)

cbow_model = CBOW(vocab_size, embed_dim)
cbow_model.embeddings.weight.data.copy_(embeddings)
cbow_model.embeddings.weight.requires_grad = False  


# -----------------------------
# 6. Create title embeddings
# -----------------------------
title_embeddings = []
valid_indices = []
for i, tokens in enumerate(tokenized_titles):
    token_ids = [word_to_ix[t] for t in tokens if t in word_to_ix]
    if token_ids:
        with torch.no_grad():
            vectors = cbow_model.embeddings(torch.tensor(token_ids))
            avg_vector = vectors.mean(dim=0)
        title_embeddings.append(avg_vector)
        valid_indices.append(i)

X_title = torch.stack(title_embeddings)
y = torch.tensor([scores[i] for i in valid_indices], dtype=torch.float32).unsqueeze(1)


karmas_tensor = torch.tensor([karmas[i] for i in valid_indices], dtype=torch.float32).unsqueeze(1)  # NEW
user_ids = [by[i] for i in valid_indices]  # NEW
user_karma_lookup = {user_ids[i]: karmas_tensor[i].item() for i in range(len(user_ids))}  # NEW


# -----------------------------
# 7. Process domain names
# -----------------------------
parsed_domains = []
for url in urls:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or 'unknown'
    except:
        domain = 'unknown'
    # Clean and normalize domain names
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    parsed_domains.append(domain)

le = LabelEncoder()
domain_ids = le.fit_transform(parsed_domains)
domain_ids_tensor = torch.tensor(domain_ids, dtype=torch.long)[valid_indices]
domain_vocab_size = len(le.classes_)
domain_embed_dim = 3


# -----------------------------
# 7b. Process usernames
# -----------------------------
user_le = LabelEncoder()
user_ids_encoded = user_le.fit_transform(user_ids)
user_ids_tensor = torch.tensor(user_ids_encoded, dtype=torch.long)

user_vocab_size = len(user_le.classes_)
user_embed_dim = 4  # chosen based on typical embedding heuristics


# -----------------------------
# 8. Regression Model
# -----------------------------


class UpvotePredictor(nn.Module):
    def __init__(self, title_embed_dim, domain_vocab_size, domain_embed_dim, users_vocab_size, users_embed_dim):
        super().__init__()
        self.domain_embedding = nn.Embedding(domain_vocab_size, domain_embed_dim)
        self.user_embedding = nn.Embedding(users_vocab_size, users_embed_dim)
        self.model = nn.Sequential(
            nn.Linear(title_embed_dim + domain_embed_dim + users_embed_dim + 1, 128),  # CHANGED
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, title_embed, domain_id, user_id, karma):  # CHANGED
        domain_vec = self.domain_embedding(domain_id)
        user_vec = self.user_embedding(user_id)  # NEW
        x = torch.cat([title_embed, domain_vec, user_vec, karma], dim=1)  # CHANGED
        return self.model(x)



model = UpvotePredictor(embed_dim, domain_vocab_size, domain_embed_dim, user_vocab_size, user_embed_dim)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 


# -----------------------------
# 9. Dataset and Training
# -----------------------------
class HNDataset(Dataset):
    def __init__(self, title_embeds, domain_ids, user_ids, karmas, labels):  # CHANGED
        self.title_embeds = title_embeds
        self.domain_ids = domain_ids
        self.user_ids = user_ids # NEW
        self.karmas = karmas  # NEW
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.title_embeds[idx], self.domain_ids[idx], self.user_ids[idx], self.karmas[idx], self.labels[idx]  # CHANGED

dataset = HNDataset(X_title, domain_ids, user_ids, karmas, y)  # CHANGED
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Training regression model...")
for epoch in range(10):
    total_loss = 0
    for title_embed, domain_id, user_id, karma, label in dataloader:  # CHANGED
        pred = model(title_embed, domain_id, user_id, karma)  # CHANGED
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    ### Uncomment to activate learning rate schedular ###
    #scheduler.step()
    #print(f"Epoch {epoch:2d} — LR: {scheduler.get_last_lr()[0]:.4f}")



# -----------------------------
# 10. Prediction Function
# -----------------------------
def predict_upvotes(title, url, user_id):  # CHANGED
    tokens = preprocess(title)
    token_ids = [word_to_ix.get(t) for t in tokens if t in word_to_ix]
    token_ids = [i for i in token_ids if i is not None]
    if not token_ids:
        return None

    with torch.no_grad():
        vectors = cbow_model.embeddings(torch.tensor(token_ids))
        avg_embed = vectors.mean(dim=0)

    try:
        parsed = urlparse(url)
        domain = parsed.netloc or 'unknown'
        
    except:
        domain = 'unknown'

    if domain.startswith("www."):
        domain = domain[4:]

    try:
        domain_id = le.transform([domain])[0]
    except:
        domain_id = 0  # fallback

    try:
        user_enc = user_le.transform([user_id])[0]
    except:
        user_enc = 0 # fallback


    domain_tensor = torch.tensor([domain_id], dtype=torch.long)
    user_tensor = torch.tensor([user_enc], dtype=torch.long)
    karma_value = user_karma_lookup.get(user_id, 0)  # NEW
    karma_tensor = torch.tensor([[karma_value]], dtype=torch.float32)  # NEW

    model.eval()
    with torch.no_grad():
        prediction = model(avg_embed.unsqueeze(0), domain_tensor, user_tensor, karma_tensor).item()  # CHANGED
    prediction = int(round(prediction, 0))  # Round to nearest integer
    return max(prediction, 1)  # Ensure at least 1 upvote


# -----------------------------
# 11. Save the model
# -----------------------------

torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": total_loss,
}, "upvote_predictor_full_0.7.pt")

print(f"Model saved to {save_path}")