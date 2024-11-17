import numpy as np
import sys
from random import random
from language_model import * 

def generate_ngrams(tokens, N):
    ngrams = zip(*[tokens[i:] for i in range(N)])
    return [tuple(ngram) for ngram in ngrams]


def get_next_word_probabilities(context, lm_type, model, unigrams, bigrams, trigrams, lambda_values, N, k):
    context_tokens = [token for token in context.split() if token.isalpha()]
    context_ngram = tuple(context_tokens[-(N-1):]) if len(context_tokens) >= N-1 else tuple(context_tokens)
    next_word_probs = {}
    possible_next_words = unigrams.keys()

    if lm_type == 'g':
        for word in possible_next_words:
            test_ngram = context_ngram + (word,)
            # Check if the n-gram exists for the 'g' model
            if N == 3 and (test_ngram not in trigrams or context_ngram not in bigrams):
                continue  # Skiping if trigram or preceding bigram is OOD
            elif N == 2 and test_ngram not in bigrams:
                continue  # Skiping if bigram is OOD
            prob_val = model.get(test_ngram, 0)  # Retrieving the smoothed probability
            if prob_val > 0:
                next_word_probs[word] = prob_val

    elif lm_type == 'i':
        for word in possible_next_words:
            bigram = context_ngram[-1:] + (word,) if len(context_tokens)>0 else (word,)
            trigram = context_ngram + (word,) if len(context_ngram) == N-1 else (word,)
            # Checking if the n-gram exists for interpolation for OOD
            if not (bigram in bigrams or trigram in trigrams):
                continue  # Skiping if bigram or trigram is OOD
            interpolated_prob = compute_interpolated_prob((word,), bigram, trigram, unigrams, bigrams, trigrams, lambda_values)
            next_word_probs[word] = interpolated_prob


    elif lm_type == 'w':
        for word in possible_next_words:
            if N == 1:
                next_word_probs[word] = unigrams.get(word, 0) / sum(unigrams.values())
            elif N == 2:
                bigram = context_ngram + (word,)
                if bigram in bigrams:
                    next_word_probs[word] = bigrams[bigram] / unigrams.get(context_ngram[-1], sum(unigrams.values()))
            elif N == 3:
                trigram = context_ngram + (word,)
                if trigram in trigrams:
                    next_word_probs[word] = trigrams[trigram] / bigrams.get(context_ngram, sum(bigrams.values()))

    sorted_predictions = sorted(next_word_probs.items(), key=lambda x: x[1], reverse=True)[:k]
    return {word: prob for word, prob in sorted_predictions}


def main(lm_type, corpus_path, k):
    corpus = read_corpus(corpus_path)
    tokens = tokenize(corpus)
    tokens = remove_punctuation(tokens)
    N = 3  

    flat_tokens = flatten(tokens)
    unigrams = generate_unigrams(flat_tokens)
    bigrams = generate_bigrams(flat_tokens)
    trigrams = generate_trigrams(flat_tokens)

    model = None
    lambda_values = None
    if lm_type == 'g':
        ngrams = generate_ngrams(flat_tokens, N)
        model = good_turing_smoothing(ngrams)
    elif lm_type == 'i':
        lambda_values = compute_lambdas(unigrams, bigrams, trigrams)
    elif lm_type == 'w':
        model = {'unigrams': unigrams, 'bigrams': bigrams, 'trigrams': trigrams}

    input_sentence = input("Input sentence: ")
    next_word_probs = get_next_word_probabilities(input_sentence, lm_type, model, unigrams, bigrams, trigrams, lambda_values, N, k)

    if not next_word_probs:
        print("Out of context")
    else:
        for word, prob in next_word_probs.items():
            print(f"{word}: {prob}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python gen.py <lm_type> <corpus_path> <k>")
        sys.exit(1)

    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])

    main(lm_type, corpus_path, k)
