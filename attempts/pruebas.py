import torch
import matplotlib.pyplot as plt
import json
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from string import punctuation
from transformers import BertTokenizer, BertModel

model = BertModel.from_pretrained('bert-base-uncased')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

with open("articles.json", "r") as f:
    data = json.load(f)

stopwords_es = set(stopwords.words('spanish'))  # Análisis de frecuencia de las palabras sin stopwords
#patron = r'[a-zA-Z]+'  # patron para eliminado de caracteres especiales, números, símbolos y espacios en blanco.
#tokenizer = RegexpTokenizer(patron)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for id, dictt in data.items():
    words = tokenizer.tokenize(data[id]["text"].lower())
    filtered_tokens = [token for token in words if token not in stopwords_es and len(token) > 2 and (token) not in punctuation]
    for index, word in enumerate(filtered_tokens):
        filtered_tokens[index] = " ".join(filtered_tokens[index])
    print(filtered_tokens)
    print("\n" * 3)

    tokenized_text = tokenizer.tokenize(data[id]["text"].lower())
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    max_seq_length = 512
    tokenized_text = tokenized_text[:max_seq_length]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    encoded_layers, _ = model(tokens_tensor)

    token_i = 0

    # Crear una lista de vectores
    vectors = [encoded_layers[layer_i][batch_i][i] for i in range(token_i, len(tokenized_text))]

    # Apilar los vectores en un tensor  
    token_embeddings = torch.stack(vectors)

    
    with torch.no_grad():
        outputs = model(tokens_tensor)
        encoded_layers = outputs[0]

    # Obtener la puntuación de cada token
    layer_i = 0
    batch_i = 0
    token_i = 0

    # Obtener el vector de características del token
    encoded_layers, _ = model(tokens_tensor)
    token_embeddings = torch.stack(encoded_layers[layer_i][batch_i][token_i:])

    # Sumar las capas para obtener la representación de la palabra
    summed_token_embeddings = torch.sum(token_embeddings, dim=0)

    # Convertir el tensor en una matriz de numpy para facilitar la manipulación
    summed_token_embeddings = summed_token_embeddings.numpy()

    # Calcular la puntuación de cada palabra
    word_scores = {}
    for i, word in enumerate(tokenized_text):
        word_scores[word] = summed_token_embeddings[i]

    print(word_scores)


"""for id_del_articulo in data_articulos:   
    id_articulos.append(id_del_articulo)
     for texto_articulo in [id_del_articulo]["text"]:

           texto_articulos.append(texto_articulo)

print(id_articulos)"""
      

#print(texto_articulos) 

"""

for id_del_articulo in articulos:   
    id_articulos.append(id_del_articulo)
    articulo = articulos[id_del_articulo]
    

#    texto = articulos['text']
#    textos_articulos.append(texto)

# Buscar las keywords y sus scores:


nlp = spacy.load('es_core_news_sm') 

keywords = {}

for texto in textos_articulos:
    doc = nlp(texto)
    for token in doc:
        if token.is_alpha and not token.is_stop:
            if token.text in keywords:
                keywords[token.text] += token.similarity(doc)
            else:
                keywords[token.text] = token.similarity(doc)

# Enlazar los artículos con los vídeos: Iteramos la lista de diccionarios de vídeos y comparar 
# los keywords de cada vídeo con los keywords de cada artículo para encontrar la mejor coincidencia.

for video in videos:
    keywords_video = set(video['keywords'])
    mejor_score = 0
    mejor_articulo = None

    for articulo in articulos:
        keywords_articulo = set(articulo['keywords'])
        score = sum([keywords[token] for token in keywords_video & keywords_articulo])

        if score > mejor_score:
            mejor_score = score
            mejor_articulo = articulo

    video['mejor_articulo'] = mejor_articulo['titulo']

print(textos_articulos)
print(keywords)
"""

