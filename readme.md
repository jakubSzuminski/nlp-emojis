# NLP for proposing emojis
The goal of the model: *given a sentence, output an emoji that fits this sentence best*.

### Example
Input for the model: 
> "Let's go play some football."

Output:
> "Let's go play some football ⚽"

This model uses pre-trained word embeddings from Kaggle: https://www.kaggle.com/datasets/watts2/glove6b50dtxt?resource=download

# Neural Network Architecture
This model is an RNN - Recurrent Neural Network.

It's created with `tensorflow.keras` and is based on the following layers: 
- Embedding 
- LSTM (long short term memory)
- Dropout
- Softmax (for the output layer of the emoticon)

For each word, the structure is this:
- embedding -> LSTM -> dropout (0.5 probability) -> LSTM

After $x^{<T_x>}$ we go through an additional Dropout layer followed by a Softmax layer.

# Dataset
The dataset consists of (X, Y) where:
- X contains of 127 sentences
- Y contains of 127 integers (value between 0 and 4 corresponding to an emoji for each sentence)

0 - ❤️
1 - ⚾
2 - 😊
3 - 😞
4 - 🍴

# Jupyter Notebook
If you want to see how the model works, open the Jupyter Notebook file (.ipynb) in an environment that has the basic Python libraries for Data Science installed:
- pandas
- numpy
- tensorflow
- matplotlib
- csv
- emoji

This task is from the DeepLearning.AI course on Coursera: https://www.coursera.org/learn/nlp-sequence-models