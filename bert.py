import json
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Cargar modelo y tokenizer de BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Establecer parámetro max_length
max_length = 512

def obtener_keywords(texto):
    # Limpiar texto y tokenizar
    texto_limpio = limpiar_texto(texto)
    tokens = tokenizer.encode(texto_limpio, add_special_tokens=True, max_length=max_length)

    # Pasar tokens al modelo
    input_ids = torch.tensor([tokens])
    outputs = model(input_ids)

    # Obtener los logits de la última capa y calcular el score
    logits = outputs[0].detach().numpy()[0]
    scores = softmax(logits)

    # Obtener las palabras más relevantes
    palabras = tokenizer.convert_ids_to_tokens(tokens)
    keywords = [(palabras[i], scores[i]) for i in range(len(palabras)) if scores[i] > 0.5]

    return keywords

def limpiar_texto(texto):
    # Aquí implementa la limpieza de texto que desees, como eliminar stopwords o caracteres especiales
    texto_limpio = texto.lower()
    return texto_limpio

# Función para calcular la softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Cargar datos de archivo JSON
with open('articles.json') as f:
    datos = json.load(f)

# Obtener keywords para cada texto en los datos
for data in datos:
    keywords = obtener_keywords(data['text'])
    print(keywords)

print(limpiar_texto)
