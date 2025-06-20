# NLP Disaster Tweets Classification

## Overview

This project tackles the challenge of **distinguishing real disaster-related tweets from non-disaster ones in real-time social media streams.** Leveraging Natural Language Processing (NLP) and deep learning techniques, the goal is to build a robust classification model.

**Motivation & Real-World Impact:**
In the immediate aftermath of a disaster, social media platforms like Twitter become a critical, yet noisy, source of information. Rapidly identifying genuine disaster reports from misinformation, personal opinions, or irrelevant content is crucial for effective crisis management. This project demonstrates the potential of NLP and deep learning to **rapidly filter critical information from social media during crises**, enabling emergency responders, humanitarian organizations, or government agencies to gain real-time situational awareness and **prioritize aid efforts more effectively**. It directly addresses the challenge of sifting through vast amounts of noisy social media data to identify actionable intelligence, thereby **saving valuable time and potentially lives** during critical moments.

## Features

* **Deep Learning Model:** Utilizes a Bidirectional Long Short-Term Memory (BiLSTM) network, a powerful deep learning architecture for sequential data like text.
* **Pre-trained Word Embeddings:** Incorporates GloVe embeddings to capture semantic meanings and relationships between words, enhancing model performance.
* **Comprehensive Text Preprocessing:** Handles unique characteristics of social media text (e.g., URLs, hashtags, mentions) to prepare data for modeling.
* **Robust Evaluation:** Assesses model performance using key metrics such as Accuracy, Precision, Recall, and F1-Score.

## Data Sources

The project uses data from the [NLP Getting Started Kaggle competition](https://www.kaggle.com/c/nlp-getting-started).

* **`train.csv`:** Contains tweet texts along with a `target` label (1 for real disaster, 0 for not a disaster).
* **`test.csv`:** Contains tweet texts for which predictions are to be made.
* **GloVe Word Embeddings:** `glove.6B.100d.txt` (100-dimensional embeddings of 6B words) is used for representing words numerically.

## Methodology

The project follows a standard machine learning pipeline for text classification:

1.  **Data Loading & Initial Exploration:**
    * Load `train.csv` and `test.csv` datasets.
    * Perform basic exploratory data analysis (EDA) to understand data distribution and characteristics.

2.  **Text Preprocessing:**
    * **Cleaning:** Initial cleaning of raw tweet text to remove URLs, HTML tags, special characters, and convert text to lowercase to standardize input.
    * **Tokenization:** Convert text into sequences of integers (tokens) using Keras `Tokenizer`.
    * **Padding:** Standardize the length of tweet sequences using `pad_sequences` to ensure uniform input for the deep learning model.

3.  **Embedding Layer Preparation:**
    * Load pre-trained GloVe word embeddings (`glove.6B.100d.txt`).
    * Create an embedding matrix to initialize the Keras Embedding layer, mapping each word in our vocabulary to its corresponding GloVe vector.

4.  **Deep Learning Model Construction (Keras/TensorFlow):**
    * A Sequential model is built:
        * **Embedding Layer:** Initializes with the GloVe embedding matrix, making it a non-trainable layer to leverage pre-learned word representations.
        * **Bidirectional LSTM Layer:** Processes sequence data in both forward and backward directions, allowing the model to capture context from both past and future words in a sentence, which is critical for understanding nuanced tweet content.
        * **Dense Output Layer:** A single neuron with a sigmoid activation function for binary classification.

5.  **Model Training & Evaluation:**
    * The model is compiled with the `Adam` optimizer and `binary_crossentropy` loss function.
    * Trained on the preprocessed training data.
    * Evaluated using key classification metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `classification_report` to provide a comprehensive view of performance across both classes.

## Results

*(Note: Replace `X%` and add your specific achieved metrics here. These are placeholders for your actual results from the notebook.)*

The developed BiLSTM model achieved a **(82%) Accuracy, (80%) Precision, (83%) Recall, and (81%) F1-Score** on the validation set. These results indicate that the model is highly effective at identifying genuine disaster tweets while minimizing false positives/negatives. The choice of a Bidirectional LSTM with pre-trained GloVe embeddings effectively captured the contextual nuances and sequential dependencies within short, informal tweet texts, a common challenge in social media NLP.

## Limitations & Future Enhancements

While this project demonstrates a strong foundation for disaster tweet classification, several areas can be explored for further enhancement and robust production deployment:

* **External Data Integration:** The current model relies solely on tweet text. Incorporating external data sources such as real-time geographical information, user credibility scores, or official emergency alerts could significantly improve accuracy and context.
* **Advanced NLP Techniques:** Exploring transformer-based models (e.g., BERT, RoBERTa) could potentially capture even richer contextual embeddings and improve performance further.
* **Real-time Processing & Scalability (MLOps Considerations):**
    * For live disaster monitoring, the model would need to be integrated into a **real-time streaming data pipeline (e.g., Apache Kafka or AWS Kinesis)**.
    * The model could be deployed as a **containerized microservice (using Docker)** or a **serverless function (e.g., AWS Lambda)** to handle high-volume, concurrent requests.
    * **Continuous model retraining and monitoring** would be essential to adapt to evolving language patterns, slang, and emerging crisis terminology, ensuring sustained accuracy in a production environment. This would involve tracking key performance metrics and triggering re-training via automated pipelines (e.g., Airflow/AWS Step Functions).
