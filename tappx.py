
import json     #libreria para cargar los datos de  archivos JSON
import spacy    


# Cargar los datos del archivo de artículos de prensa

with open('articles.json', 'r') as f:
    data_articulos = json.load(f)

# Cargar los datos del archivo de vídeos

with open('videos.json', 'r') as f:
    videos = json.load(f)

# Almacenamos los id de los artículos
# Almacenamos los textos de los artículos antes de limpiar

id_articulos = []  
texto_articulos = [] 

for id_del_articulo in data_articulos:   
    id_articulos.append(id_del_articulo)
    for texto_articulo in [data_articulos]:
        texto_articulos.append(texto_articulo)

print(id_articulos)
      

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
