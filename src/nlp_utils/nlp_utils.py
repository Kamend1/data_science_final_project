from transformers import logging as hf_logging
import numpy as np


def classify_long_transcript(text, tokenizer, nlp):
    """
    This method receives a text, a tokenizer, and an nlp model.
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

    chunks = [tokens[i:i + max_chunk_size] for i in range(0, len(tokens), max_chunk_size)]

    results = []
    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk)
        res = nlp(chunk_text, truncation=True, max_length=512)
        results.append(res[0])

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