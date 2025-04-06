# Thesis Recommendation System

This project is a Flask-based web application that provides thesis recommendations based on user-provided keywords, selected approaches, and work types. It uses various machine learning models for embedding and similarity computation.

## Features

- **Keyword-based Recommendations**: Enter keywords to get relevant thesis topics.
- **Model Selection**: Choose from multiple models like TF-IDF, BERT, SBERT, RoBERTa, DistilBERT, ALBERT, and XLNet.
- **Pagination**: Navigate through recommendations using "Previous" and "Next" buttons.
- **Language Support**: Supports Czech (`CZ`) and English (`EN`) data processing.

## Technologies Used

- **Backend**: Flask, PyTorch, scikit-learn, Hugging Face Transformers
- **Frontend**: HTML, CSS, JavaScript (jQuery)
- **Data Processing**: Pandas, Sentence Transformers
- **Deployment**: ngrok (for development), Google Cloud Run (for production)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Im0ne/thesis-recsys.git
    cd thesis-recommendation-system
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    python bp.py
    ```
