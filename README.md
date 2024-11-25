# Word2Vec
For our final project, we were curious about how word embeddings are learned and trained, so we set out to implement the word2vec algorithm called skip-gram with negative sampling that is discussed in chapter 6 of Speech and Language Processing (3rd ed. draft) by Dan Jurafsky and James H. Martin for our naive implementation. Specifically this implementation requires only the numpy library.
## The Dataset
For training, we used two datasets. One was based on Discord messages that we have decided to not make public for privacy reasons. There were more than a million such messages. The other dataset is 25,000 reviews from [Stanford's Large Movie Review Dataset](http://www.aclweb.org/anthology/P11-1015).

We decided to use two datasets for a couple reasons. First, the dataset we gathered from Discord is noisy as there is not necessarily continuity between messages, possibly leading to worse embeddings. Second, the movie review dataset is filled with polarizing reviews, giving our word vectors a good avenue to learn sentiment based embeddings, indicating whether our results were successful.
## Usage
To get the required libraries, users should run `pip3 install -r requirements.txt`

To train word embeddings on our naive model, a user should run `python3 train_word2vec.py <model_name> <dataset_file_or_directory>`. Users can optionally run python3 train_word2vec.py -h to see a list of changeable hyperparameters to fine-tune their models.

To train our model on a database like that of the movie reviews, all reviews in question should be placed in a single directory stored as simple text documents, all other file structure should be eliminated.

After the model is trained, users should run the command `python3 load_model.py` and enter the name of their model when prompted. Users can then query the most similar words within a model.

To see the results of automatic evaluation, users should run `python3 compare_models.py <naive_model_directory> <gensim_model_directory> <training_dataset>` and the program will output the comparison. This file takes in hyperparameters through the same method as train_word2vec.py.

## AI Acknoledgement
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
Discussion regarding appropriate hyperparameters and training corpus: https://chatgpt.com/share/6740bee0-80ec-8008-8eb8-30b69c5ae1f6