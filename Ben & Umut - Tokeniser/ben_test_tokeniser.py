import psycopg2
import torch
from torch.nn.utils.rnn import pad_sequence
import re
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer


# Connect to PostgreSQL and load titles
def load_titles(pg_url, limit=10000):
    conn = psycopg2.connect(pg_url)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT title FROM hacker_news.items
        WHERE title IS NOT NULL
        LIMIT {limit}
    """)
    titles = [row[0] for row in cur.fetchall()]
    conn.close()
    return titles

stemmer = PorterStemmer()
# Basic tokenizer using regex
def tokenize(text):
    '''
    Keep punctuation as separate tokens
    Lowercase and split on word boundaries
    This regex captures words and punctuation separately
    \w+ matches word characters, [^\w\s] matches punctuation
    '''
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    
    '''
    Stem tokens if they are alphabetic
    Use Porter Stemmer for stemming
    If the token is not alphabetic, keep it as is
    This will stem words like "running" to "run", but keep numbers and punctuation intact
    "Hello!" → Tokens: ["hello", "!"]
    "!?" → Tokens: ["!", "?"]
    '''
    
    return [stemmer.stem(token) if token.isalpha() else token 
            for token in tokens]



# Yield token lists for each title
def yield_tokens(texts):
    for text in texts:
        yield tokenize(text)

# Manually build vocab (no torchtext)
def build_vocab_from_iterator(iterator, specials=['<pad>', '<unk>']):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    # Index-to-string list
    itos = specials + [token for token, _ in counter.items()]
    
    # String-to-index dict with default to <unk>
    stoi = defaultdict(lambda: specials.index('<unk>'), {token: idx for idx, token in enumerate(itos)})

    # Return a callable vocab object similar to torchtext
    class Vocab:
        def __init__(self, stoi, itos):
            self.stoi = stoi
            self.itos = itos
        def __call__(self, tokens):
            return [self.stoi[token] for token in tokens]
        def __getitem__(self, token):
            return self.stoi[token]
        def __len__(self):
            return len(self.itos)

    return Vocab(stoi, itos)

# Convert tokenized titles to padded tensor
def encode_titles(titles, vocab):
    token_ids = [
        torch.tensor(vocab(tokenize(title)), dtype=torch.long)
        for title in titles
    ]
    return pad_sequence(token_ids, batch_first=True, padding_value=vocab['<pad>'])

# Main pipeline
def main(pg_url):
    titles = load_titles(pg_url)
    vocab = build_vocab_from_iterator(yield_tokens(titles), specials=['<pad>', '<unk>'])
    encoded_tensor = encode_titles(titles, vocab)

    print("Vocab size:", len(vocab))
    print("Example tensor shape:", encoded_tensor.shape)
    print("First encoded title:", encoded_tensor[0])
    print(tokenize("This is a test! With punctuation?"))
    # Output: ['thi', 'is', 'a', 'test', '!', 'with', 'punctuat', '?']
    return encoded_tensor, vocab

if __name__ == "__main__":
    pg_url = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
    encoded_titles, vocab = main(pg_url)
