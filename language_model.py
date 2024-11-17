import numpy as np
from collections import Counter
from scipy.stats import linregress
from random import sample, seed
import sys
from tokenizer import tokenize, generate_ngrams

roll_number = '2023201047'

def read_corpus(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def remove_punctuation(tokens):
    return [[token for token in sentence if token.isalpha()] for sentence in tokens]

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def generate_unigrams(tokens):
    return Counter(tokens)

def generate_bigrams(tokens):
    return Counter(generate_ngrams(tokens, 2))

def generate_trigrams(tokens):
    return Counter(generate_ngrams(tokens, 3))

def good_turing_smoothing(ngrams, k=3):
    N = sum(Counter(ngrams).values())  # Total number of N-grams
    C = Counter(ngrams)  # Frequency of each N-gram
    N_c = Counter(C.values())  # Frequency of frequencies

    # Calculation  of Z_r.
    sorted_counts = sorted(set(N_c.keys()))
    Z_r = {}
    for i, r in enumerate(sorted_counts):
        q = sorted_counts[i - 1] if i > 0 else 0
        t = sorted_counts[i + 1] if i < len(sorted_counts) - 1 else r
        if N_c[r] == 0 or t == q:
            continue
        Z_r[r] = N_c[r] / ((t - q) / 2 if r != t else t - q)

    # Log-log linear regression to get parameters.
    log_r = [np.log(r) for r in sorted_counts if r <= k and r in Z_r]
    log_Z_r = [np.log(Z_r[r]) for r in sorted_counts if r <= k and r in Z_r]
    slope, intercept, _, _, _ = linregress(log_r, log_Z_r)

    smoothed_probs = {}
    for ngram, r in C.items():
        if r <= k:
            S_Nr = np.exp(intercept + slope * np.log(r))
            smoothed_prob = (r + 1) * S_Nr / N_c[r] / N
        else:
            smoothed_prob = r / N
        smoothed_probs[ngram] = smoothed_prob

    #Handling Probabilities for unseen N-grams
    p0 = N_c[1] / N if 1 in N_c else 1 / N
    smoothed_probs['<unseen>'] = p0

    return smoothed_probs

def calculate_sentence_probability(sentence, model, N):
    sentence_tokens = tokenize(sentence)
    sentence_ngrams = generate_ngrams(sentence_tokens[0], N)
    probability = 1.0
    for ngram in sentence_ngrams:
        probability *= model.get(ngram, model['<unseen>'])

    return max(probability,1e-50) 


def compute_lambdas(unigrams, bigrams, trigrams):
    lambda1_increment = 0.0
    lambda2_increment = 0.0
    lambda3_increment = 0.0

    for trigram, trigram_count in trigrams.items():
        if trigram_count > 0:
            (t1, t2, t3) = trigram
            bigram_t1_t2 = (t1,t2)
            bigram_t2_t3=(t2,t3)

            unigram_t3 = t3
            unigram_t2 =  t2

            bigram_t1_t2_count = bigrams.get(bigram_t1_t2, 0)
            bigram_t2_t3_count=bigrams.get(bigram_t2_t3,0)
            unigram_t3_count = unigrams.get(unigram_t3, 0)
            unigram_t2_count = unigrams.get(unigram_t2, 0)

            N=sum(unigrams.values())

            if bigram_t1_t2_count > 1:
                a=(trigram_count-1)/(bigram_t1_t2_count -1)
            else:
                a = 0
            
            if unigram_t2_count > 1:
                b=(bigram_t2_t3_count-1)/(unigram_t2_count-1)
            else:
                b = 0
            
            c=(unigram_t3_count-1)/(N-1)
            
            max_inc = max(a, b, c)
            if max_inc == a:
                lambda3_increment += trigram_count
            elif max_inc == b:
                lambda2_increment += trigram_count
            elif max_inc == c:
                lambda1_increment += trigram_count

    total = lambda1_increment + lambda2_increment + lambda3_increment
    if total > 0:
        lambda1 = lambda1_increment / total
        lambda2 = lambda2_increment / total
        lambda3 = lambda3_increment / total
    else:
        lambda1, lambda2, lambda3 = 0.33, 0.33, 0.34 #assigning default probabilities as backup

    # print(lambda1, lambsda2, lambda3)
    return [lambda1, lambda2, lambda3]

def compute_interpolated_prob(unigram, bigram, trigram, unigrams, bigrams, trigrams, lambda_values):
    min_prob = 1e-5 # Defining a minimum probability to avoid log(0)
    total_unigrams = sum(unigrams.values())
    unigram_prob = max(unigrams.get(unigram, 0) / total_unigrams, min_prob)
    bigram_prob = max(bigrams.get(bigram, 0) / unigrams.get(bigram[0], 1), min_prob)
    trigram_prob = max(trigrams.get(trigram, 0) / bigrams.get(trigram[:2], 1), min_prob)
    
    interpolated_prob = lambda_values[0] * unigram_prob + lambda_values[1] * bigram_prob + lambda_values[2] * trigram_prob
    return interpolated_prob  

def compute_interpolated_sentence_prob(sentence, unigrams, bigrams, trigrams, lambda_values):

    tokens = 2*['<BOS>'] + tokenize(sentence)[0] + ['<EOS>']  # Tokenizing sentence and adding BOS/EOS tags
    probability = 1.0
    for i in range(2, len(tokens)):
        unigram = (tokens[i-1],)
        bigram = (tokens[i-1], tokens[i])
        trigram = (tokens[i-2], tokens[i-1], tokens[i])
        prob = compute_interpolated_prob(unigram, bigram, trigram, unigrams, bigrams, trigrams, lambda_values)
        probability *= prob 
    return max(probability,1e-50)


def perplexity_helper(sentences, lm_type, model, unigrams, bigrams, trigrams, lambda_values, N):
    perplexities = []
    for sentence_tokens in sentences:
        s=sentence_tokens
        sentence_tokens = 2*['<BOS>'] + sentence_tokens + ['<EOS>']

        if lm_type == 'g':
            probs = calculate_sentence_probability(' '.join(sentence_tokens), model, N)
            perplexities.append((1 / probs) ** (1/int(len(s) + 1)))

        elif lm_type == 'i':
            probs = compute_interpolated_sentence_prob(' '.join(sentence_tokens), unigrams, bigrams, trigrams, lambda_values)
            perplexities.append((1 / probs) ** (1/int(len(s) + 1)))

    return perplexities


def compute_perplexity(corpus_path, lm_type):
    corpus = read_corpus(corpus_path)
    tokens = tokenize(corpus)  
    tokens = remove_punctuation(tokens) 
    N = 3

    model, unigrams, bigrams, trigrams, lambda_values = None, None, None, None, None

    if lm_type == 'g':
        flat_tokens = flatten(tokens)
        ngrams = generate_ngrams(flat_tokens, N)
        model = good_turing_smoothing(ngrams)

    elif lm_type == 'i':
        flat_tokens = flatten(tokens)
        unigrams = generate_unigrams(flat_tokens)
        bigrams = generate_bigrams(flat_tokens)
        trigrams = generate_trigrams(flat_tokens)
        lambda_values = compute_lambdas(unigrams, bigrams, trigrams)

    seed(42)  # Ensuring reproducibility
    test_indices = sample(range(len(tokens)), 1000)
    test_set = [tokens[i] for i in test_indices]
    train_set = [tokens[i] for i in range(len(tokens)) if i not in test_indices]

    train_perplexities = perplexity_helper(train_set, lm_type, model, unigrams, bigrams, trigrams, lambda_values, N)
    test_perplexities = perplexity_helper(test_set, lm_type, model, unigrams, bigrams, trigrams, lambda_values, N)

    for set_name, set_perplexities in [("train", train_perplexities), ("test", test_perplexities)]:
        # avg_perplexity = (np.mean([x for x in set_perplexities if x < 1e20]))
        avg_perplexity =np.exp(np.mean(np.log((set_perplexities))))
        output_filename = f"{roll_number}_LM{lm_type}_{set_name}-perplexity.txt"
        with open(output_filename, 'w') as f_out:
            f_out.write(f"{avg_perplexity}\n")
            for sentence, perplexity in zip((train_set if set_name == "train" else test_set), set_perplexities):
                f_out.write(f"{' '.join(sentence)}\t{perplexity}\n")
        print(f"Perplexity scores written to {output_filename}")


def main(lm_type, corpus_path):
    corpus = read_corpus(corpus_path)
    tokens = tokenize(corpus) 

    
    tokens = remove_punctuation(tokens) # Remove non-alphabetic tokens

    N = 3  # Assuming a trigram model
    processed_tokens = [['<BOS>']*(N-1) + token_list + ['<EOS>'] for token_list in tokens]
    flat_processed_tokens = flatten(processed_tokens)
 
    if lm_type == 'g':
        ngrams = generate_ngrams(flat_processed_tokens, N)
        model = good_turing_smoothing(ngrams)
    elif lm_type == 'i':
        unigrams = generate_unigrams(flat_processed_tokens)
        bigrams = generate_bigrams(flat_processed_tokens)
        trigrams = generate_trigrams(flat_processed_tokens)
        lambda_values = compute_lambdas(unigrams, bigrams, trigrams)
    else:
        raise ValueError("Invalid LM type. Use 'g' for Good-Turing or 'i' for Interpolation.")
    
    sentence = input("Input sentence: ")
    sentence_tokens = tokenize(sentence)  # Tokenizing it into a list of tokenized sentences
    sentence_tokens = remove_punctuation(sentence_tokens)[0]  # Filtering and getitng the first sentence's tokens

    
    if lm_type == 'g':
        probability = calculate_sentence_probability(' '.join(sentence_tokens), model, N)
        print(f"Score: {probability}")
    elif lm_type == 'i':
        probability = compute_interpolated_sentence_prob(' '.join(sentence_tokens), unigrams, bigrams, trigrams, lambda_values)
        print(f"Score: {probability}")

    # compute_perplexity(corpus_path, lm_type)
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python language_model.py <lm_type> <corpus_path>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])