import psycopg2
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import re


# a module to tokenize either titles 
# from a PostgreSQL database containing Hacker News items

# connect to PostgreSQL and load titles
# loads titles into as a list of stings where each element in the 
# list is a single title from the database
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

# split the title into 
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

