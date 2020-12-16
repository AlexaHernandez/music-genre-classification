import pandas as pd
import transformers as trfmrs
from os.path import dirname, abspath, join
import numpy as np
import torch
import pickle

"""This module is to compute the DistilBERT embedding features."""


ROOT = dirname(dirname(abspath(__file__)))
DATA = join(ROOT, "data", "scraped-lyrics-v2-preprocessed.csv")


def truncate(lyrics):
    if len(lyrics) <= 510:
        return lyrics
    return middle_tokens(lyrics)


def middle_tokens(lyrics):
    # If length of lyrics is even
    if len(lyrics) % 2 == 0:
        right_middle = len(lyrics) // 2
        left_middle = right_middle - 1
        start = max(left_middle - 254, 0)
        end = min(right_middle + 255, len(lyrics))
    # If length of lyrics is odd
    else:
        middle = len(lyrics) // 2
        start = max(middle - 255, 0)
        end = min(middle + 255, len(lyrics))
    return lyrics[start: end]


def get_max_length(tokenized_vals):
    max = 0
    for i in tokenized_vals:
        if len(i) > max:
            max = len(i)
    return max


def get_bert_embeddings(lyrics):
    trunc_lyrics = [truncate(x) for x in lyrics]

    # Load DistilBERT tokenizer and model
    model_class, tokenizer_class, pretrained_weights = (
    trfmrs.DistilBertModel, trfmrs.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Tokenize truncated lyrics
    tokenized = [tokenizer.encode(x, add_special_tokens=True) for x in trunc_lyrics]

    # Pad lyrics so that they are all the same size
    max_length = get_max_length(tokenized)
    padded = np.array([i + [0] * (max_length - len(i)) for i in tokenized])

    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded).to(torch.int64)
    attention_mask = torch.tensor(attention_mask).to(torch.int64)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    return features


if __name__ == "__main__":
    """Sample usage."""
    df = pd.read_csv(DATA)
    lyrics = df.lyrics.values
    batches = np.array_split(lyrics, 100)

    for i in range(len(batches)):
        if i == 0:
            distil_bert_features = get_bert_embeddings(batches[i])
        else:
            features = get_bert_embeddings(batches[i])
            distil_bert_features = np.concatenate((distil_bert_features, features), axis=0)

    print(type(distil_bert_features))
    print(f"Computed distilBERT features with shape: {distil_bert_features.shape}")
    print(f"Sample: {distil_bert_features[:5]}")

    outfile = open("distil_bert_embeddings.pickle", 'wb')
    pickle.dump(distil_bert_features, outfile)
    outfile.close()
