import torch
import matplotlib.pyplot as plt
import json
import nltk
from base64 import encode
from cgitb import text
from lib2to3.pgen2 import tokenize
from pyexpat import model
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


stopwords_es = set(stopwords.words('spanish'))       # Análisis de frecuencia de las palabras sin stopwords
patron = r'[a-zA-Z]+'                                # patron para eliminado de caracteres especiales, números, símbolos y espacios en blanco.
tokenizer = RegexpTokenizer(patron)

for id, dictt in data.items():
    words = tokenizer.tokenize(data[id]["text"].lower())
    filtered_tokens = [token for token in words if token not in stopwords_es and len(token) > 2 and (token) not in punctuation]
    #for index, word in enumerate(filtered_tokens):
     #   filtered_tokens[index] = " ".join(filtered_tokens[index])
    print(filtered_tokens)
    print("\n"*3)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_text = tokenizer.tokenize(data[id]["text"].lower())

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens])

encoded_layers, _ = model(tokens_tensor)

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
for i, word in enumerate(text):
    word_scores[word] = summed_token_embeddings[i]

print(indexed_tokens)


#data[id]["text"] = filtered_tokens
#freq_dist = nltk.FreqDist(words)

################################################################################################

# ya veremos si terminamos usando lo siguiente:


"""freq_dist = FreqDist(filtered_tokens)
print(freq_dist.most_common(22))

#print(type(freq_dist))

freq_dist.plot(22, cumulative=False)
plt.show()"""