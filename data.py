import gzip
import json
import random
import requests
from transformers import DistilBertTokenizerFast
from utils import label2id

def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head: break
    return random.sample(reviews, min(sample_size, len(reviews)))

def prepare_data(model_name='distilbert-base-cased', sample_per_genre=1000):
    genre_url_dict = {
        'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
        'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
        'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
        'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
        'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
        'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
        'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
        'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
    }
    
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    
    for genre, url in genre_url_dict.items():
        reviews = load_reviews(url, head=10000, sample_size=sample_per_genre)
        split = int(len(reviews) * 0.8)
        train_texts.extend(reviews[:split])
        train_labels.extend([label2id[genre]] * split)
        test_texts.extend(reviews[split:])
        test_labels.extend([label2id[genre]] * (len(reviews) - split))
        
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    
    return train_encodings, train_labels, test_encodings, test_labels
