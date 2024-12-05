# sub-event-detection

## Overview
The goal of this challenge is to apply machine learning and artificial intelligence techniques to a real-world binary classification problem. The task involves building a model to predict the occurrence of notable events within one-minute intervals of football matches using a dataset derived from tweets from the 2010 and 2014 FIFA World Cup tournaments.

### Dataset
- Each one-minute interval is annotated with a binary label:
  - `0`: No notable event occurred.
  - `1`: A significant event occurred (e.g., goal, penalty, red card, etc.).
- Labels are assigned only if the interval aligns closely with the event time.

---

## Preprocessing

The dataset underwent a series of preprocessing steps to ensure its utility for modeling:
1. **Duplicate Removal:** All duplicate tweets and retweets (marked as `RT`) were removed, accounting for over 50% of the dataset.
2. **Exclusion of Mentions:** Tweets with mentions (`@username`) were excluded due to their minimal relevance.
3. **Short Tweet Prioritization:** Tweets with fewer than 10 words were prioritized for their higher relevance to events.
4. **Emoji and Link Removal:** Emojis and links were removed to align with the pre-trained models' training data.
5. **Concatenation:** Tweets within each time period were concatenated to form a single input text.

---

## Approaches

### Bag of Words (BoW)
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency) was used to extract features.
- **Vocabulary:** Included football-specific terms (e.g., `goal`, `penalty`) and the top 30 most frequent tokens.
- **Classifier:** Models like Logistic Regression, Random Forest, and MLP were trained with 10-fold cross-validation for robustness.

### Encoder-Only Transformers
- **Model:** A LongFormer model was fine-tuned, capable of handling inputs up to 4096 tokens.
- **Fine-Tuning Strategy:** Selective freezing of layers (final two layers and classifier layer trainable) and a dynamic learning rate scheduler were used.
- **Objective:** Minimized binary cross-entropy loss, with the best model chosen based on maximum minimum accuracy across validation matches.

### Decoder-Only Transformers
- **Model:** Llama 3.1 8B Instruct was used for in-context learning, generating summaries for short tweets.
- **Strengths:** High interpretability and rich contextual summaries.
- **Limitations:** Complexity of the model and limited fine-tuning options.
- **Note:** This model was used solely for inference.

---

## Code Structure

- **`src/`:** Contains Python modules for data processing and modeling.
- **`notebooks/`:** Includes experiments and results.
- **`scripts/`:** Python scripts for executing tasks.
