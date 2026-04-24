from transformers import logging as hf_logging
import numpy as np
import re
import html


def classify_long_transcript_batched(text, tokenizer, nlp, batch_size=16):
    """
    This method receives a text, a tokenizer, and a nlp model.
    It splits the text in chunks to accommodate the tokenizer limit.
    Rates each chunk and appends the result to a list.
    """

    hf_logging.set_verbosity_error()

    max_chunk_size = 500

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False
    )

    token_chunks = [tokens[i:i + max_chunk_size] for i in range(0, len(tokens), max_chunk_size)]

    string_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

    results = nlp(string_chunks, batch_size=batch_size, truncation=True)

    return results


def aggregate_sentiment(results):
    """
    Converts a list of chunk results into a single aggregate score.
    Positive = 1, Neutral = 0, Negative = -1
    """
    scores = []
    for res in results:
        label = res['label']
        confidence = res['score']

        if label == 'positive':
            scores.append(confidence)
        elif label == 'negative':
            scores.append(-confidence)
        else:
            scores.append(0)

    return np.mean(scores)


def clean_for_finbert(text):
    """
    Decodes HTML, removes tables, and normalizes whitespace.
    """
    if not text:
        return ""

    text = html.unescape(text)

    text = re.sub(r'(\d+[\d,.]*\s+){3,}', ' ', text)

    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'Item\s\d\.\d\d', '', text, flags=re.IGNORECASE)

    return text.strip()