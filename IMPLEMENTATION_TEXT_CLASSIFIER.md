# Implementation: Text Classifier

## Overview

The text classifier detects AI-generated text using a hybrid approach combining a pre-trained language model (DistilBERT) with normalized handcrafted linguistic features. This architecture leverages both deep semantic understanding and explicit statistical patterns that distinguish human from AI writing.

## Development Progression

### Iteration 1: TF-IDF + Logistic Regression
The initial approach used traditional TF-IDF vectorization with logistic regression. This model performed poorly, failing to capture the nuanced differences between human and AI text beyond simple word frequency patterns.

### Iteration 2: Feature-Based Classifier
The second iteration extracted 20+ handcrafted linguistic features (perplexity, burstiness, paragraph length, punctuation density, etc.) and trained a neural network classifier. However, this model suffered from critical flaws:
- **Unnormalized features**: Features with different scales dominated training
- **Overfitting to paragraph length**: The model learned to classify primarily based on text length rather than linguistic quality
- Poor generalization to real-world text

### Iteration 3: Hybrid DistilBERT + Normalized Features (Current)
The current model addresses previous failures by:
- Normalizing all handcrafted features to comparable scales
- Combining DistilBERT embeddings with normalized features
- Balancing deep semantic understanding with explicit statistical signals

**Architecture:**
```
Input Text
    ↓
    ├─→ DistilBERT Tokenizer → DistilBERT (frozen) → [CLS] embedding (768-dim) ─┐
    │                                                                            │
    └─→ Feature Extractor → Normalized features (20-dim) ────────────────────────┤
                                                                                 ↓
                                                        Concatenate [768 + 20 = 788-dim]
                                                                                 ↓
                                                        Dense(256) + Dropout(0.3) + ReLU
                                                                                 ↓
                                                        Dense(64) + Dropout(0.2) + ReLU
                                                                                 ↓
                                                        Dense(2) → Softmax → [Human, AI]
```

**Key Features Extracted:**
- **Perplexity**: Calculated using GPT-2 small model (pre-computed for all 200k samples)
- **Burstiness**: Variance in sentence lengths
- **Paragraph statistics**: Average length, count (normalized)
- **Lexical diversity**: Type-token ratio, unique word percentage
- **Punctuation patterns**: Density, variety, ratios
- **Syntactic complexity**: Average sentence length, word length
- **Readability scores**: Flesch Reading Ease, Flesch-Kincaid Grade

## Dataset Construction

**Total: 200,000 samples**

**Source 1: artem9k/ai-text-detection-pile (114,569 samples)**
- Mix of Reddit WritingPrompts, OpenAI Webtext, NY Times articles
- AI text from GPT-2, GPT-3, GPT-4, Gemma, Mistral

**Source 2: HC3 Dataset (85,431 samples)**
- reddit_eli5: 51,336 human + 16,660 ChatGPT
- finance: 3,933 human + 4,503 ChatGPT
- open_qa: 1,187 human + 3,546 ChatGPT
- medicine: 1,248 human + 1,334 ChatGPT
- wiki_csai: 842 human + 842 ChatGPT

**Label Balance:**
- Human: 115,885 (57.9%)
- AI: 84,115 (42.1%)

The dataset provides diverse domains (social media, news, technical writing, conversational Q&A) and multiple AI model generations, ensuring the classifier generalizes across different text types and generation methods.

## Training Configuration

**Hardware:**
- AMD Radeon RX 7900 XTX (24GB VRAM)
- DirectML backend (Windows)
- Separate training workstation

**Training Process:**
1. **Perplexity Pre-computation**: Calculated perplexity scores for all 200k samples using GPT-2 small model
2. **Feature Normalization**: Standardized all handcrafted features (zero mean, unit variance)
3. **Model Training**: 
   - Optimizer: AdamW
   - Learning rate: 2e-5 (DistilBERT layers frozen, only head trained)
   - Batch size: 32
   - Training in progress

**Rationale for Hybrid Approach:**
- Pure LLM approaches risk overfitting to training data distribution
- Handcrafted features provide interpretable signals (perplexity, burstiness)
- Normalized features prevent the paragraph length overfitting observed in Iteration 2
- Frozen DistilBERT reduces computational cost while maintaining semantic understanding
- Combination captures both "what is said" (semantics) and "how it's said" (style)

## Implementation Details

**Feature Normalization Strategy:**
All features are standardized using training set statistics:
```
normalized_feature = (feature - mean) / std
```

This ensures features like perplexity (range: 0-1000+) and punctuation density (range: 0-1) contribute equally to classification decisions.

**Perplexity Calculation:**
Perplexity measures how "surprised" a language model is by the text. Lower perplexity indicates more predictable text (typical of AI-generated content, which follows learned patterns closely), while higher perplexity suggests more creative or unexpected language choices (typical of human writing). Calculated as:
```
perplexity = exp(cross_entropy_loss)
```

Using GPT-2 small model with a sliding window approach:
- Window size: 1024 tokens
- Stride: 256 tokens (768-token overlap between windows)
- Each window scores tokens from position 256 onward to avoid edge effects

All 200,000 samples had perplexity pre-computed before training to avoid repeated inference overhead during epochs.

## Key Challenges Addressed

1. **Feature Scale Imbalance**: Solved by standardization
2. **Overfitting to Spurious Correlations**: Hybrid approach prevents reliance on single features
3. **Computational Efficiency**: Frozen DistilBERT backbone reduces training time
4. **Domain Generalization**: Diverse dataset spanning multiple text types and AI models

## Integration

The trained model is deployed in the browser extension's native messaging host, processing text submissions in real-time. The hybrid architecture provides both a classification decision and confidence score, allowing users to assess reliability.
