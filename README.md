# 🎬 IMDB Movie Review Sentiment Analysis using CNN

This project implements a **Convolutional Neural Network (CNN)** model for binary sentiment classification of movie reviews from the **IMDB dataset**. It uses **text preprocessing, tokenization, word embeddings**, and a **1D CNN architecture** to classify reviews as **positive** or **negative**.

---

## 📌 Dataset

- **Source**: [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size Used**: 5000 randomly sampled reviews for faster training
- **Columns**: `review` (text), `sentiment` (positive/negative)

---

## 📊 Model Architecture

- **Tokenizer**: Keras `Tokenizer` with padding
- **Embedding Layer**: 100-dimensional embeddings
- **Conv1D Layer**: 128 filters with kernel size 5 and ReLU activation
- **Global Max Pooling**: Downsamples features
- **Dense Layers**:
  - 10 units with ReLU
  - 1 unit with Sigmoid (binary classification)

---

## 🛠️ Libraries Used

- `TensorFlow / Keras`
- `scikit-learn`
- `pandas`, `numpy`, `matplotlib`
- `nltk` (for stopwords and stemming)
- `pickle` (to save tokenizer)

---

## 🧹 Text Preprocessing

1. HTML decoding using `html.unescape`
2. HTML tag removal using regex
3. Lowercasing
4. Non-letter removal
5. Stopword removal using NLTK
6. Stemming using `PorterStemmer`

---

## 📈 Training Performance

- Trained for 3 epochs with a batch size of 64
- Accuracy and loss are visualized for both training and validation sets

---

## 🚀 Prediction Function

A simple function to predict sentiment:

```python
predict_sentiment("Good movie", tokenizer, model, maxlen)
# Output: "Positive"
