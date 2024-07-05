import nltk
from nltk.corpus import brown
import numpy as np
from collections import defaultdict

# Ensure the Brown corpus is downloaded
nltk.download('brown')

# Preprocess the corpus
sentences = brown.sents()
corpus = [[word.lower() for word in sentence] for sentence in sentences]

# Build the vocabulary and create word to index mapping
word_counts = defaultdict(int)
for sentence in corpus:
    for word in sentence:
        word_counts[word] += 1

vocab = {word: i for i, (word, _) in enumerate(word_counts.items())}
index_to_word = {i: word for word, i in vocab.items()}
vocab_size = len(vocab)

# Hyperparameters
embedding_dim = 100
window_size = 2
learning_rate = 0.01
epochs = 10

# Initialize weights
W1 = np.random.randn(vocab_size, embedding_dim)
W2 = np.random.randn(embedding_dim, vocab_size)

# Training the Word2Vec model
for epoch in range(epochs):
    loss = 0
    for sentence in corpus:
        for i, word in enumerate(sentence):
            context_indices = list(range(max(0, i - window_size), min(len(sentence), i + window_size + 1)))
            context_indices.remove(i)
            context_words = [sentence[j] for j in context_indices]
            target_word = word
            target_index = vocab[target_word]
            context_indices = [vocab[word] for word in context_words if word in vocab]

            # Forward pass
            h = W1[target_index]
            u = np.dot(W2.T, h)
            y_pred = np.exp(u) / np.sum(np.exp(u))

            # Calculate loss
            for context_index in context_indices:
                loss -= np.log(y_pred[context_index])

            # Backward pass
            e = y_pred
            e[context_indices] -= 1

            dW2 = np.outer(h, e)
            dW1 = np.dot(W2, e)

            W1[target_index] -= learning_rate * dW1
            W2 -= learning_rate * dW2

    print(f'Epoch: {epoch}, Loss: {loss}')

# Retrieve the vector for the word "king"
king_vector = W1[vocab['king']]

# Find words similar to "king"
similarities = np.dot(W1, king_vector)
sorted_indices = np.argsort(similarities)[::-1]
similar_words = [index_to_word[i] for i in sorted_indices[:6] if i != vocab['king']]

print(f'Top 5 words similar to "king": {similar_words}')
