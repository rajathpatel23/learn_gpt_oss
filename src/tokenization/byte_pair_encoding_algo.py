from collections import defaultdict, Counter

def get_stats(corpus):
    """Count frequency of all symbol pairs in the corpus"""
    pairs = Counter()
    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    print(pairs)
    return pairs

def merge_vocab(pair, corpus):
    """Merge the most frequent pair in the corpus"""
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    new_corpus = {}
    for word, freq in corpus.items():
        new_word = word.replace(bigram, replacement)
        new_corpus[new_word] = freq
    return new_corpus

# Example corpus
corpus = {
    "l o w </w>": 5,
    "l o w e s t </w>": 2,
    "n e w e r </w>": 6,
    "w i d e r </w>": 3,
}

# Number of merges
num_merges = 30
for i in range(num_merges):
    pairs = get_stats(corpus)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    corpus = merge_vocab(best, corpus)
    print(f"Step {i+1}: merge {best}")
    print("Updated corpus:", corpus)
    print()
