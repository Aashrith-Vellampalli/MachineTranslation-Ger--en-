# Machine Translation: Seq2Seq vs Transformer (German → English)

This project implements **Neural Machine Translation (NMT)** models **from scratch** for **German → English** translation using the **Multi30k dataset**.  
The focus is on understanding **sequence modeling, attention mechanisms, decoding strategies, and evaluation**, rather than relying on pre-built libraries.

---

## Models Implemented

### Seq2Seq with Attention (LSTM)
- Encoder–Decoder architecture using **LSTM**
- **Luong-style attention**
- Teacher forcing during training
- Greedy decoding and Beam Search
- Successfully trained and evaluated

**Performance**
- **BLEU score (test set): ~27**
- Produces fluent and semantically meaningful translations

---

### Transformer (From Scratch)
- Multi-Head Self-Attention
- Positional embeddings
- Encoder–Decoder stack
- Causal and padding masks
- Noam learning rate scheduler
- Greedy and Beam Search decoding

**Status**
- Training loss decreases normally
- Suffers from **token repetition during decoding**
- **Very low BLEU score**
- Included to demonstrate practical difficulties of training Transformers from scratch on small datasets

---
## 📊 Evaluation

- BLEU score computed using **NLTK corpus BLEU**
- Includes smoothing
- Evaluated on both training and test sets

```python
from nltk.translate.bleu_score import corpus_bleu
