### Tokenization: From Text to Integers

Language models are mathematical functions; they operate on numbers, not raw text. Tokenization is the crucial first step in converting human-readable text into a sequence of integers (tokens) that a model can process. These tokens are then mapped to embedding vectors, as described in `gpt_oss_architecture.md`.

---

### 1. Naive Approaches and Their Flaws

#### Word-Level Tokenization
The most intuitive approach: split text by spaces and punctuation.
*   **Problems**:
    1.  **Vocabulary Explosion**: A language like English has hundreds of thousands of words. The model's vocabulary would be enormous, making the final embedding and output layers computationally massive.
    2.  **Out-of-Vocabulary (OOV) Words**: If the model encounters a word not seen during training (e.g., a new slang term, a typo, or a technical name), it has no token for it. It typically maps it to an `<UNK>` (unknown) token, losing all semantic meaning.
    3.  **Poor Generalization**: The model treats `eat`, `eating`, and `eaten` as three completely separate, unrelated tokens. It fails to capture the shared root `eat`, making it harder to learn morphological relationships.

#### Character-Level Tokenization
The opposite extreme: split text into individual characters.
*   **Advantage**: The vocabulary is tiny and fixed (e.g., ~256 for bytes). There are no OOV words.
*   **Problems**:
    1.  **Loss of Semantic Meaning**: A single character like `t` has very little semantic value on its own. The model must expend significant capacity just to learn the structure of common words from characters.
    2.  **Long Sequences**: A simple sentence becomes a very long sequence of tokens. This makes it computationally difficult for the model to learn long-range dependencies and can quickly exceed the context window.

---

### 2. The Solution: Subword Tokenization

Modern LLMs use a hybrid approach called **subword tokenization**, which combines the best of both worlds.

*   **Core Idea**: Common words are kept as single, unique tokens (e.g., `the`, `is`, `and`), while rare or complex words are broken down into smaller, meaningful sub-units (e.g., `tokenization` -> `token` + `ization`).
*   **Benefits**:
    *   It gracefully handles OOV words by breaking them into known subwords.
    *   It captures morphological relationships (e.g., `eating` -> `eat` + `ing`).
    *   It keeps the vocabulary size manageable while maintaining semantic richness.

#### Algorithm: Byte-Pair Encoding (BPE)
BPE is a data-driven algorithm used to create a subword vocabulary. It starts with a base vocabulary of individual characters (or bytes) and learns merge rules from a large text corpus.

1.  **Initialization**: The initial vocabulary consists of all single bytes (0-255).
2.  **Iteration**:
    a. Count the frequency of all adjacent token pairs in the training corpus.
    b. Find the most frequent pair (e.g., `('e', 'r')`).
    c. **Merge** this pair into a new, single token (`'er'`).
    d. Add this new token to the vocabulary and replace all occurrences of the pair in the corpus with the new token.
3.  **Repeat**: Continue this process for a predetermined number of merges, which defines the final vocabulary size.

The `byte_pair_encoding_algo.py` script in this repository provides a simple, hands-on example of this process.

---

### 3. Modern Tokenizers in GPT Models

OpenAI's models use a library called `tiktoken` for high-performance BPE tokenization.

*   **`cl100k_base`**: The tokenizer used for models like GPT-3.5 and GPT-4. It has a vocabulary size of ~100,000.
*   **`o200k_base`**: A newer tokenizer for more advanced models. It has a larger vocabulary of ~200,000 and includes special tokens designed for chat and instruction-following.

#### Special Tokens for Chat

Modern tokenizers reserve special tokens to encode conversational structure, which is critical for chat models.

*   **Chat Formatting**: Tokens like `<|im_start|>` and `<|im_end|>` are used to delineate turns in a conversation, allowing the model to understand the flow of a multi-speaker dialogue.
*   **Role-Based Prompting**: Tokens are used to specify roles (e.g., `system`, `user`, `assistant`), preserving the instructions and context for how the model should behave.

These special tokens are not part of the regular text but are essential metadata that the model uses to understand its task. As shown in `tokenizer.py`, you can enable them with `allowed_special="all"`.




