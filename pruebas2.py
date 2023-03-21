import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

with open('articles.json', 'r') as f:
    data_articulos = json.load(f)

with open('videos.json', 'r') as f:
    data_videos = json.load(f)

data_articulos = pd.DataFrame(data_articulos).transpose()
data_videos = pd.DataFrame(data_videos).transpose()

def extraer_texto(dataframe, columns):
    texto = []
    for col in columns:
        if col in dataframe.columns:
            texto += dataframe[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x)).tolist()
    return texto

id_articulos = data_articulos.index.tolist()
title_articulos = extraer_texto(data_articulos, ['title'])
keywords_articulos = extraer_texto(data_articulos, ['keywords'])
class_articulos = extraer_texto(data_articulos, ['categoriaIAB'])
text_articulos = extraer_texto(data_articulos, ['text'])
url_articulos = extraer_texto(data_articulos, ['url'])

id_videos = data_videos.index.tolist()
keywords_videos = extraer_texto(data_videos, ['keywords'])
class_videos = extraer_texto(data_videos, ['categoriaIAB'])
text_videos = extraer_texto(data_videos, ['text'])
url_videos = extraer_texto(data_videos, ['url'])

# Preprocesamiento de texto
stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convertir a minúsculas
    text = text.lower()

    # Tokenizar el texto
    tokens = nltk.word_tokenize(text, language='spanish')

    # Eliminar stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Aplicar stemming y lemmatization
    #tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens]    #esta linea causa errores.?¿
    #print(tokens)
    return tokens

filtered_tokens = []
for text in text_videos:
    tokens = preprocess_text(text)
    filtered_tokens += tokens



# Obtener scores de keywords en videos
keyword_scores = {}
for keyword in keywords_articulos:
    keyword_tokens = preprocess_text(keyword)
    keyword_scores[keyword] = 0
    for video_text in text_videos:
        video_tokens = preprocess_text(video_text)
        total_words = len(video_tokens)
        for token in video_tokens:
            if token in keyword_tokens:
                keyword_scores[keyword] += 1 / total_words
    #print(keyword_scores)

# Obtener las mejores relaciones entre artículos y videos
relaciones = {}
for id_articulo in id_articulos:
    id_videos_relacionados = []
    for id_video in id_videos:
        video_score = 0
        #print(video_score)
        for keyword in keywords_articulos:
            if keyword in keyword_scores:
                video_keywords = keywords_videos[id_videos.index(id_video)]
                if keyword in video_keywords:
                    video_score += keyword_scores[keyword]
        if video_score > 0:
            id_videos_relacionados.append(id_video)
    relaciones[id_articulo] = id_videos_relacionados

    if id_videos_relacionados:
        print(f"Los vídeos que se relacionan mejor con tu artículo son:", ', '.join(id_videos_relacionados))
