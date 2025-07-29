# 🤖 Code Comment Generator with Transformers

## 📋 Project Overview

**Complete Transformer implementation from scratch for automatic Python code documentation**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-green.svg)](https://streamlit.io)

> *Built from scratch following "Attention Is All You Need" - demonstrates complete ML engineering pipeline and critical evaluation of model limitations.*

## 🏗️ Technical Architecture

### Transformer Model (1.39M Parameters)

* 🔄 **4 Encoder + 4 Decoder layers** with **8-head self-attention**
* 🧠 **Custom tokenizer** with 181-token vocabulary (built from scratch)
* ⚙️ Fully from scratch implementation of:

  * Multi-head attention
  * Scaled dot-product attention
  * Positional encoding
  * Layer normalization
  * Residual connections

### Training Results

* 🔻 **Loss reduced from 5.24 → 1.65** over 90 epochs
* 🧪 Trained on 15 curated high-quality Python code-comment pairs

---

## 📁 Project Structure

```
src/
├── models/              # Transformer model implementation
│   ├── attention.py     # Multi-head attention
│   ├── encoder.py       # Encoder layers
│   ├── decoder.py       # Decoder layers
│   └── transformer.py   # Combined Transformer model
├── data_processing/     # Tokenization, dataset utilities
├── training/            # Training loop, checkpoints, early stopping
└── inference/           # Streamlit interface + model loading
```

---

## 🚧 Challenges & Solutions

### 🧩 Challenge 1: Data Scarcity & Collection

* **Problem**: GitHub API rate limits + parsing complexity
* **Solution**: Hand-curated 15 diverse examples to demonstrate concept efficacy

### ⚙️ Challenge 2: Attention Mechanism Complexity

* **Problem**: Implementing and debugging tensor shapes
* **Solution**: Built modular attention layers with careful broadcasting

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

### 🧪 Challenge 3: Overfitting

* **Problem**: High training accuracy but poor generalization
* **Solution**: Early stopping, regularization, gradient clipping

---

## 📊 Results & Insights

| Metric          | Value                                  |
| --------------- | -------------------------------------- |
| Training Loss   | 5.24 → 1.65                            |
| Validation Loss | \~4.6 (expected due to low data)       |
| Strength        | Works well on simple utility functions |
| Limitation      | Fails on complex multi-branch logic    |

> 💡 **Key Insight**: Model complexity isn't a substitute for good data. High-quality, diverse training data is vital.

---

## 💼 Real-World Learnings

| Limitation                   | Reason               | Suggested Fix                                |
| ---------------------------- | -------------------- | -------------------------------------------- |
| Poor complex code generation | Low data             | Use CodeBERT or CodeT5 for transfer learning |
| Overfit training set         | Small dataset        | Automate collection from GitHub repos        |
| Output quality               | Narrow code patterns | Augment dataset, filter noisy examples       |

---

## 💻 Quickstart

### 🔧 Setup

```bash
git clone https://github.com/VarenyaVisen/code-comment-generator.git
cd code-comment-generator
pip install -r requirements.txt
streamlit run src/inference/streamlit_app.py
```

---

## 🛠️ Tech Stack

* **Python** - Core language
* **PyTorch** - Model training
* **Streamlit** - Web interface
* **NumPy, Matplotlib** - Utilities & debugging
* **Custom Tokenizer** - Built from scratch

---

## 🎓 Skills Demonstrated

* ✅ Deep Learning: Transformer internals & attention math
* ✅ ML Engineering: Training loops, monitoring, checkpoints
* ✅ Web Dev: Deployable interface using Streamlit
* ✅ Software Design: Modular, reusable architecture
* ✅ Honest Evaluation: Understanding when models fail

---

## 🔮 Future Improvements

* 🔍 **Expand Dataset**: Scrape and clean 1K+ GitHub examples for better generalization
* 📈 **Model Optimization**: Try distilled transformer variants to reduce inference time
* 🧠 **Pretrained Code Models**: Fine-tune CodeBERT or CodeT5 for improved performance
* 🌐 **VSCode Plugin**: Wrap model into a local extension for real-time comment suggestions
* 🚀 **REST API Deployment**: Dockerize and serve via FastAPI or Flask for broader integration

---

## 👤 Author

**Varenya Visen**
📫 Email: [varenyavisen@gmail.com](mailto:varenyavisen@gmail.com)
🌐 LinkedIn: [@VarenyaVisen](https://www.linkedin.com/in/varenya-visen-a2680b265/)
💻 GitHub: [VarenyaVisen](https://github.com/VarenyaVisen)

---

⭐ *Star this repo if you found it helpful*!
*Built from scratch with ❤️ and curiosity for deep learning + real-world ML!*
