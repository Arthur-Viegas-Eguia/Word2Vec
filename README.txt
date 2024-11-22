No code was written by any AI like ChatGPT, but clarifying questions were asked like:

Q: When computing gradients for word2vec training, should you normalize the embeddings after changing them?

A: No, you generally should **not normalize the embeddings** during or after each update when training word2vec models. Here's why:

### 1. **Embedding Training Depends on Magnitude**
   - Word2Vec (and similar models) optimizes the embeddings using techniques like stochastic gradient descent (SGD). The magnitude of the embedding vectors is part of the learned representation and contributes to how similar or dissimilar words are in the embedding space.
   - Normalizing embeddings after each update would restrict their magnitude and interfere with the training process, preventing the model from learning meaningful differences in scale that could represent important relationships.

### 2. **Normalization is Optional Post-Training**
   - Once training is complete, you might normalize the embeddings (e.g., to unit length) for specific tasks like cosine similarity computations. Normalization ensures that the similarity metric is purely based on the direction of vectors rather than their magnitudes.

### 3. **Negative Sampling and Softmax Depend on Unnormalized Vectors**
   - Techniques like **negative sampling** or **hierarchical softmax**, often used in word2vec, rely on the dot product between word vectors and context vectors. These dot products naturally account for the magnitude of vectors, so normalizing them during training would alter the optimization dynamics.

### 4. **When Normalization Could Be Used**
   - In some specialized applications (e.g., when training embeddings to have specific properties), normalization may be used during training. However, this is not standard practice in word2vec.

### In Summary:
- **During Training**: Do not normalize embeddings after updates.
- **Post-Training**: Normalize if needed for specific downstream tasks or when using metrics like cosine similarity.

Let me know if you'd like further clarification or a deeper dive into any of the training processes!

AI conversations:
Discussion regarding appropriate learning rates: https://chatgpt.com/share/6740bee0-80ec-8008-8eb8-30b69c5ae1f6