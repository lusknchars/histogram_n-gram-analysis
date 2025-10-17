# 🎵 Language Modeling with Histogram N-Gram Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-green.svg)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

*Building Intelligent Text Generation Models from Scratch*

[📊 Overview](#overview) • [🎯 Models](#models) • [📈 Results](#results) • [🚀 Usage](#usage)

</div>

---

## 🎯 Overview

This project implements **statistical language models** using N-Gram analysis to understand and generate text. Using pop song lyrics as training data, the models learn linguistic patterns and word distributions to predict and generate coherent sequences.

### ✨ Key Features

- 📚 **Three N-Gram Models**: Unigram, Bigram, and Trigram implementations
- 🔍 **Statistical Analysis**: Word frequency distributions and conditional probabilities
- 🎵 **Text Generation**: Automatic lyric generation based on learned patterns
- 📊 **Probability Calculations**: Complete mathematical framework for predictions
- 🛠️ **NLTK Integration**: Leveraging industry-standard NLP toolkit

---

## 📊 Dataset

**Source:** Pop Song Lyrics  
**Example:** "Never Gonna Give You Up" by Rick Astley

**Processing Steps:**
- Tokenization using NLTK
- Punctuation removal
- Lowercase normalization
- Special character filtering

**Statistics:**
- **Total Tokens:** Variable based on input
- **Vocabulary Size:** Unique words in corpus
- **N-Gram Types:** Unigram, Bigram, Trigram

---

## 🏗️ Models Implemented

### 1️⃣ **Unigram Model**

The simplest language model that considers each word independently.

**Mathematical Foundation:**
```
P(W) = Count(W) / Total Words
```

**Characteristics:**
- ✅ Simple probability calculation
- ✅ Fast computation
- ❌ No context awareness
- ❌ Poor text generation quality

**Example:**
```
P("never") = Count("never") / Total Words
```

---

### 2️⃣ **Bigram Model** ⭐

Models word sequences by considering one previous word as context.

**Mathematical Foundation:**
```
P(W_t | W_{t-1}) = Count(W_{t-1}, W_t) / Count(W_{t-1})
```

**Characteristics:**
- ✅ Context-aware (1 word)
- ✅ Better predictions than unigram
- ✅ Reasonable generation quality
- ⚠️ Limited context window

**Example:**
```
P("gonna" | "never") = Count("never gonna") / Count("never")
```

---

### 3️⃣ **Trigram Model** ⭐⭐

Advanced model considering two previous words as context.

**Mathematical Foundation:**
```
P(W_t | W_{t-2}, W_{t-1}) = Count(W_{t-2}, W_{t-1}, W_t) / Count(W_{t-2}, W_{t-1})
```

**Characteristics:**
- ✅ Extended context (2 words)
- ✅ Best prediction accuracy
- ✅ More coherent text generation
- ⚠️ Requires more training data

**Example:**
```
P("give" | "never", "gonna") = Count("never gonna give") / Count("never gonna")
```

---

## 📐 Implementation Details

### **Frequency Distribution**

Using NLTK's `FreqDist` to calculate word frequencies:

```python
from nltk import FreqDist
fdist = FreqDist(tokens)
# Returns: {'never': 36, 'gonna': 36, 'give': 12, ...}
```

### **Conditional Probability**

For bigrams:
```python
P(word_t | word_{t-1}) = freq_bigrams[(word_{t-1}, word_t)] / fdist[word_{t-1}]
```

### **Text Generation**

Two approaches implemented:
1. **Sequential Generation**: Use each token from corpus
2. **Recursive Generation**: Use model's own output as next input

---

## 📈 Results

### **Unigram Model Performance**

```
✅ Computation: Fast
❌ Coherence: Very Low
❌ Context: None
```

**Example Output:**
```
"never never never give you gonna..."
```
*Repetitive and lacks context*

---

### **Bigram Model Performance**

```
✅ Computation: Fast
✅ Coherence: Medium
✅ Context: 1 word
```

**Example Output:**
```
"never gonna give you up never gonna let you down..."
```
*Recognizable patterns emerge*

---

### **Trigram Model Performance** ⭐

```
✅ Computation: Moderate
✅ Coherence: High
✅ Context: 2 words
✅ Quality: Best
```

**Example Output:**
```
"never gonna give you up never gonna let you down never gonna run around and desert you..."
```
*Most coherent and contextually appropriate*

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
NLTK 3.8+
NumPy
Pandas
Matplotlib
```

### Installation

```bash
# Clone the repository
git clone https://github.com/luixnchxs/language-modeling-ngrams.git
cd language-modeling-ngrams

# Install dependencies
pip install nltk numpy pandas matplotlib

# Download NLTK data
python -m nltk.downloader punkt
```

### Quick Start

```python
import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize

# 1. Load and preprocess text
text = "your song lyrics here"
tokens = preprocess(text)

# 2. Create frequency distributions
fdist = FreqDist(tokens)
freq_bigrams = FreqDist(nltk.bigrams(tokens))
freq_trigrams = FreqDist(nltk.trigrams(tokens))

# 3. Make predictions
predictions = make_predictions("never gonna", freq_trigrams)
print(predictions[0])  # Most likely next word
```

---

## 📊 Model Comparison

| Model | Context Size | Accuracy | Speed | Coherence |
|-------|--------------|----------|-------|-----------|
| Unigram | 0 | Low | ⚡⚡⚡ | ⭐ |
| Bigram | 1 | Medium | ⚡⚡ | ⭐⭐⭐ |
| Trigram | 2 | High | ⚡ | ⭐⭐⭐⭐⭐ |

---

## 💡 Key Learnings

Throughout this project, I deepened my understanding of:

### **1. Statistical Language Modeling**
- Probability distributions in natural language
- Conditional probability calculations
- Frequency analysis techniques

### **2. N-Gram Methodology**
- Trade-offs between context size and data requirements
- Markov assumption in language modeling
- Sparse data challenges

### **3. Text Generation**
- Sequential vs recursive generation strategies
- Context window importance
- Quality vs diversity trade-offs

### **4. NLP Fundamentals**
- Tokenization techniques
- Text preprocessing pipelines
- NLTK library capabilities

---

## 🔬 Mathematical Framework

### **Unigram Probability**
```
P(W₁, W₂, ..., Wₙ) = P(W₁) × P(W₂) × ... × P(Wₙ)
```

### **Bigram Probability**
```
P(W₁, W₂, ..., Wₙ) ≈ ∏ P(Wᵢ | Wᵢ₋₁)
```

### **Trigram Probability**
```
P(W₁, W₂, ..., Wₙ) ≈ ∏ P(Wᵢ | Wᵢ₋₂, Wᵢ₋₁)
```

---

## 🔮 Future Improvements

- [ ] Implement **4-gram** and **5-gram** models
- [ ] Add **smoothing techniques** (Laplace, Good-Turing)
- [ ] Integrate **back-off models** for unseen n-grams
- [ ] Compare with **neural language models** (LSTM, GPT)
- [ ] Build **interactive web interface** for text generation
- [ ] Implement **perplexity evaluation** metrics
- [ ] Add **multiple corpus support** (different music genres)
- [ ] Create **visualization dashboard** for n-gram frequencies

---

## 📚 References

- **NLTK Documentation**: [https://www.nltk.org/](https://www.nltk.org/)
- **Speech and Language Processing** by Jurafsky & Martin
- **Natural Language Processing with Python** by Bird, Klein & Loper
- **N-gram Language Models**: [Stanford NLP Course](https://web.stanford.edu/~jurafsky/slp3/)

---

## 🛠️ Technologies Used

<div align="center">

| Technology | Purpose |
|------------|---------|
| **Python** | Primary programming language |
| **NLTK** | Natural language processing toolkit |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation |
| **Matplotlib** | Data visualization |

</div>

---

## 👤 Author

**Your Name**

- 🔗 LinkedIn: [Lucas-Oliveira](https://www.linkedin.com/in/lucas-oliveira-498560246/)
- 💻 GitHub: [@lusknchars](https://github.com/lusknchars)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NLTK Team** for the comprehensive NLP toolkit
- **Rick Astley** for the memorable lyrics (training data)
- **IBM Skills Network** for the project foundation
- **NLP Community** for continuous learning resources

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star!

**Built with 💜 and Python**

*Part of the Daily ML/AI Engineering Projects Series*

[← Previous Project](../fashion_mnist_cnn) | [Next Project →](../coming-soon)

</div>

---

## 📊 Project Statistics

```
Lines of Code: ~500
Training Time: <1 minute
Models: 3 (Unigram, Bigram, Trigram)
Vocabulary Size: Variable
N-Grams Generated: Thousands
```

---

## 🐛 Known Issues

- Unigram model produces highly repetitive output
- Limited vocabulary may cause generation loops
- No handling of out-of-vocabulary words
- Basic preprocessing may miss edge cases

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Last Updated:** October 2025  
**Project #002** in the Daily ML/AI Series
