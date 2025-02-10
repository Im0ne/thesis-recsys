#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_babel import Babel, _
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (BertTokenizer, BertModel,
                          RobertaTokenizer, RobertaModel,   
                          DistilBertTokenizer, DistilBertModel,
                          AlbertTokenizer, AlbertModel,
                          XLNetTokenizer, XLNetModel)
from sentence_transformers import SentenceTransformer
from pyngrok import ngrok
import re
import time
import tracemalloc

# Load data
data = pd.read_json('./themes.json', encoding='utf-8')
data['Combined_CZ'] = data['Name_CZ'].fillna('') + " " + data['Targets_CZ'].fillna('')
data['Combined_EN'] = data['Name_EN'].fillna('') + " " + data['Targets_EN'].fillna('')

def strip_html(text):
    if isinstance(text, str):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    return ""

def load_bert(language='CZ'):
    if language == 'CZ':
        tokenizer = BertTokenizer.from_pretrained('fav-kky/FERNET-C5')
        model = BertModel.from_pretrained('fav-kky/FERNET-C5')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def load_sbert(language='CZ'):
    return SentenceTransformer('all-mpnet-base-v2')

def load_tfidf(language='CZ'):
    vectorizer = TfidfVectorizer()
    column_name = 'Combined_CZ' if language == 'CZ' else 'Combined_EN'
    vectorizer.fit(data[column_name])
    return vectorizer

def load_roberta(language='CZ'):
    if language == 'CZ':
        tokenizer = RobertaTokenizer.from_pretrained('fav-kky/FERNET-C5-RoBERTa')
        model = RobertaModel.from_pretrained('fav-kky/FERNET-C5-RoBERTa')
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
    return tokenizer, model

def load_distilbert(language='CZ'):
    if language == 'CZ':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
    else:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

def load_albert(language='CZ'):
    if language == 'CZ':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2')
    else:
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2')
    return tokenizer, model

def load_xlnet(language='CZ'):
    if language == 'CZ':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetModel.from_pretrained('xlnet-base-cased')
    else:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetModel.from_pretrained('xlnet-base-cased')
    return tokenizer, model

def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_sbert_embedding(text, model):
    return model.encode(text)

def get_tfidf_embedding(text, vectorizer):
    return vectorizer.transform([text]).toarray()[0]

def get_roberta_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_distilbert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_albert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_xlnet_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def precompute_embeddings(model_name, language):
    column_name = 'Combined_CZ' if language == 'CZ' else 'Combined_EN'
    embeddings = []
    if model_name == 'bert':
        tokenizer, model = load_bert()
        for theme in data[column_name]:
            embedding = get_bert_embedding(theme, tokenizer, model)
            embeddings.append(torch.tensor(embedding))
    elif model_name == 'sbert':
        model = load_sbert()
        for theme in data[column_name]:
            embedding = get_sbert_embedding(theme, model)
            embeddings.append(torch.tensor(embedding))
    elif model_name == 'tfidf':
        vectorizer = load_tfidf()
        vectorizer.fit(data[column_name])
        for theme in data[column_name]:
            embedding = get_tfidf_embedding(theme, vectorizer)
            embeddings.append(torch.tensor(embedding))
    elif model_name == 'roberta':
        tokenizer, model = load_roberta()
        for theme in data[column_name]:
            embedding = get_roberta_embedding(theme, tokenizer, model)
            embeddings.append(torch.tensor(embedding))
    elif model_name == 'distilbert':
        tokenizer, model = load_distilbert()
        for theme in data[column_name]:
            embedding = get_distilbert_embedding(theme, tokenizer, model)
            embeddings.append(torch.tensor(embedding))
    elif model_name == 'albert':
        tokenizer, model = load_albert()
        for theme in data[column_name]:
            embedding = get_albert_embedding(theme, tokenizer, model)
            embeddings.append(torch.tensor(embedding))
    elif model_name == 'xlnet':
        tokenizer, model = load_xlnet()
        for theme in data[column_name]:
            embedding = get_xlnet_embedding(theme, tokenizer, model)
            embeddings.append(torch.tensor(embedding))
    return torch.stack(embeddings)

# Define control flags
COMPUTE_AND_SAVE = False
LOAD_EMBEDDINGS = not COMPUTE_AND_SAVE

# Define model names and their corresponding save paths for both languages
models_list = ['tfidf', 'bert', 'sbert', 'roberta', 'distilbert', 'albert', 'xlnet']
embedding_paths_CZ = {model: f'./thesis_embeddings_{model}_CZ.pt' for model in models_list}
#embedding_paths_EN = {model: f'./thesis_embeddings_{model}_EN.pt' for model in models_list}

# Dictionary to hold loaded or computed embeddings for each language
thesis_embeddings_CZ = {}
#thesis_embeddings_EN = {}

# English version has too many nulls for themes and targets

# If computing and saving is enabled
if COMPUTE_AND_SAVE:
    for model in models_list:
        print(f"\nProcessing {model} model embeddings...")

        start_time = time.time()
        tracemalloc.start()

        # Compute embeddings for Czech and save
        thesis_embeddings_CZ[model] = precompute_embeddings(model, language='CZ')
        torch.save(thesis_embeddings_CZ[model], embedding_paths_CZ[model])

        current, peak = tracemalloc.get_traced_memory()
        end_time = time.time()

        print(f"Time taken for {model} (CZ): {end_time - start_time:.2f} seconds")
        print(f"Peak memory usage for {model} (CZ): {peak / 1024 / 1024:.2f} MB")

        #start_time = time.time()
        #tracemalloc.start()
#
        ## Compute embeddings for English and save
        #thesis_embeddings_EN[model] = precompute_embeddings(model, language='EN')
        #torch.save(thesis_embeddings_EN[model], embedding_paths_EN[model])
#
        #current, peak = tracemalloc.get_traced_memory()
        #end_time = time.time()
#
        #print(f"Time taken for {model} (EN): {end_time - start_time:.2f} seconds")
        #print(f"Peak memory usage for {model} (EN): {peak / 1024 / 1024:.2f} MB")
#
        ## Stop tracking memory
        #tracemalloc.stop()

# If loading embeddings is enabled
if LOAD_EMBEDDINGS:
    for model in models_list:
        print(f"\nLoading embeddings for {model}...")

        start_time = time.time()
        tracemalloc.start()

        # Load embeddings for Czech
        try:
            thesis_embeddings_CZ[model] = torch.load(embedding_paths_CZ[model])
        except FileNotFoundError:
            print(f"Czech embedding file for {model} not found.")

        current, peak = tracemalloc.get_traced_memory()
        end_time = time.time()
        print(f"Time taken to load {model} (CZ): {end_time - start_time:.2f} seconds")
        print(f"Peak memory usage for {model} (CZ): {peak / 1024 / 1024:.2f} MB")

        #start_time = time.time()
        #tracemalloc.start()

        # Load embeddings for English
        #try:
        #    thesis_embeddings_EN[model] = torch.load(embedding_paths_EN[model])
        #except FileNotFoundError:
        #    print(f"English embedding file for {model} not found.")

        #current, peak = tracemalloc.get_traced_memory()
        #end_time = time.time()

        #print(f"Time taken to load {model} (EN): {end_time - start_time:.2f} seconds")
        #print(f"Peak memory usage for {model} (EN): {peak / 1024 / 1024:.2f} MB")

        # Stop tracking memory
        #tracemalloc.stop()

# Load models conditionally if embeddings are loaded successfully
if thesis_embeddings_CZ: #and thesis_embeddings_EN
    models = {
        "bert": load_bert(),
        "sbert": load_sbert(),
        "tfidf": load_tfidf(),
        "roberta": load_roberta(),
        "distilbert": load_distilbert(),
        "albert": load_albert(),
        "xlnet": load_xlnet()
    }

# Combine embeddings into a final dictionary for each language if needed
embeddings_dict_CZ = {model: thesis_embeddings_CZ[model] for model in models_list if model in thesis_embeddings_CZ}
# embeddings_dict_EN = {model: thesis_embeddings_EN[model] for model in models_list if model in thesis_embeddings_EN}

app = Flask(__name__, template_folder='./templates',
                      static_folder='./static')
babel = Babel(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        keywords = request.form.get('keywords')
        approach = request.form.get('approach')
        work_type = request.form.get('workType')
        language = request.form.get('language', 'CZ')  # Default to Czech
        offset = int(request.form.get('offset', 0))

        if not keywords or not approach or not work_type:
            return jsonify({'error': 'Keywords, approach, and work type are required.'}), 400

        data_column = data['Combined_CZ']

        embedding_key = approach

        # Select the correct embeddings dictionary based on language
        thesis_embeddings = embeddings_dict_CZ.get(embedding_key) # Add check for language for english if needed

        if thesis_embeddings is None:
            print(f"Error: Embeddings for '{embedding_key}' not found in {'CZ' if language == 'CZ' else 'EN'} dictionary.")
            return jsonify({'error': f'Embeddings for "{embedding_key}" not found. Please check your embeddings files.'}), 400

        if approach in models:
            model_entry = models[approach]
            if approach == "tfidf":
                vectorizer = model_entry
                vectorizer.fit(data_column)
                user_embedding = get_tfidf_embedding(keywords, vectorizer)
            elif approach == "sbert":
                model = model_entry
                user_embedding = get_sbert_embedding(keywords, model)
            else:
                tokenizer, model = model_entry
                user_embedding = globals()[f'get_{approach}_embedding'](keywords, tokenizer, model)
        else:
            return jsonify({'error': 'Unknown approach selected.'}), 400

        # Convert user_embedding to numpy array if necessary
        if not isinstance(user_embedding, np.ndarray):
            user_embedding = np.array(user_embedding)

        similarities = cosine_similarity([user_embedding], thesis_embeddings)
        top_indices = similarities.argsort()[0][::-1]
        unique_themes = set()
        recommendations = []

        for index in top_indices:
            theme_name = data.iloc[index][f'Name_{language}']
            theme_work_type = data.iloc[index]['Type of Work']

            if theme_name not in unique_themes and theme_work_type == work_type:
                unique_themes.add(theme_name)
                recommendations.append({
                    f'Name_{language}': theme_name,
                    'Supervisor': data.iloc[index]['Supervisor'],
                    f'Targets_{language}': data.iloc[index][f'Targets_{language}']
                })

        paginated_recommendations = recommendations[offset:offset + 5]

        return jsonify(paginated_recommendations)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)