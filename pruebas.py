import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from string import punctuation
import matplotlib.pyplot as plt
import json

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

with open("articles.json", "r") as f:
    data = json.load(f)


stopwords_es = set(stopwords.words('spanish'))       # Análisis de frecuencia de las palabras sin stopwords
patron = r'[a-zA-Z]+'                                # patron para eliminado de caracteres especiales, números, símbolos y espacios en blanco.
tokenizer = RegexpTokenizer(patron)

for id, dictt in data.items():
    words = tokenizer.tokenize(data[id]["text"].lower())
    filtered_tokens = [token for token in words if token not in stopwords_es]
    #for index, word in enumerate(filtered_tokens):
     #   filtered_tokens[index] = " ".join(filtered_tokens[index])
    print(filtered_tokens)
    print("\n"*3)

#data[id]["text"] = filtered_tokens
#freq_dist = nltk.FreqDist(words)

################################################################################################

# ya veremos si terminamos usando lo siguiente:


"""freq_dist = FreqDist(filtered_tokens)
print(freq_dist.most_common(22))

#print(type(freq_dist))

freq_dist.plot(22, cumulative=False)
plt.show()"""


