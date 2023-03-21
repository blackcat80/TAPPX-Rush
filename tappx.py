import json                                                      # Libreria que usamos para manejar los datos de archivos json.
import pandas as pd                                              # Libreria Pandas para convertir a Dataframe los json.

with open('articles.json', 'r') as f:                            # Leemos el json y almacenamos los datos en data_articulos.
    data_articulos = json.load(f)

with open('videos.json', 'r') as f:                              # Leemos el json y almacenamos los datos en data_videos.
    data_videos = json.load(f)

data_articulos = pd.DataFrame(data_articulos).transpose()        # Comando pd.DataFrame para crear un dataframe con los datos de aricle.json 
#print(data_articulos)                                           # línea comentada solo por si queremos comprobar todo el contenido del json en formato dataframe.
data_videos = pd.DataFrame(data_videos).transpose()              # Lo mismo que para los videos.json 
#print(data_videos)                                              

def extraer_texto(dataframe, columns):                           # Función que nos permite iterara por cualquier dataframe.
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

#print(id_articulos)      
#print(title_articulos)
#print(keywords_articulos)
#print(class_articulos)  # pdte que solo imprima el valor, no todo. 
#print(text_articulos)
#print(url_articulos)

id_videos = data_videos.index.tolist()
keywords_videos = extraer_texto(data_videos, ['keywords'])
class_videos = extraer_texto(data_videos, ['categoriaIAB'])
text_videos = extraer_texto(data_videos, ['text'])
url_videos = extraer_texto(data_videos, ['url'])

#print(id_videos)   
#print(keywords_videos)
#print(class_videos)  # pdte que solo imprima el valor, no todo. 
#print(text_videos)
#print(url_videos)

# print(data_articulos.columns)                                  # Muestra los parametros de las columnas del Dataframe.


