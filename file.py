import pandas as pd
import json
import nltk
import pprint
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string

class File:

    def __init__(self, file):
        self.dataframe = pd.read_json(f"{file}.json").transpose()
        self.num_files = len(self.dataframe.index)
        self.id = self.dataframe.index.tolist()
        self.title = self.extraer_texto(self.dataframe, ['title'])
        self.keywords = self.extraer_texto(self.dataframe, ['keywords'])
        self.classes = self.extraer_texto(self.dataframe, ['categoriaIAB'])
        self.text = self.extraer_texto(self.dataframe, ['text'])
        self.url = self.extraer_texto(self.dataframe, ['url'])

        # To use with model https://huggingface.co/hiiamsid/sentence_similarity_spanish_es
        # self.keywords_str = []
        # for i in self.dataframe['keywords']:
        #     self.keywords_str.append((' '.join(i)))
        # self.dataframe['keywords_str'] = self.keywords_str

    def extraer_texto(self, dataframe, columns):
        texto = []
        for col in columns:
            if col in dataframe.columns:
                texto += dataframe[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x)).tolist()
        return texto

    @staticmethod
    def preprocess_text(text):
        # Preprocesamiento de texto
        stop_words = set(stopwords.words('spanish'))

        # Eliminar puntuación
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Convertir a minúsculas
        text = text.lower()
        # Tokenizar el texto
        tokens = nltk.word_tokenize(text, language='spanish')
        # Eliminar stopwords
        tokens = [token for token in tokens if token not in stop_words]
        return tokens


    # Obtener scores de keywords en un texto (Christian Version)
    def score_keywords(self, file):
        keyword_scores = {}
        for i in self.keywords:
            keyword_tokens = self.preprocess_text(i)
            for j in file.text:
                keyword_scores[self.id[index]] = 0
                video_tokens = self.preprocess_text(j)
                total_words = len(video_tokens)
                for token in video_tokens:
                    if token in keyword_tokens:
                        keyword_scores[i] += 1 / total_words
            index += 1
        print(keyword_scores)



    # Chrisitan Algorithm - (Ana iteration Version)
    def score_keywords_mix(self, file):
        """
            Iterates every text from self and gives each video a score according to that text
        """
        # For each article
        keyword_scores = {}
        for i in self.dataframe.index.tolist():
            keyword_scores[i] = {}
            # Clean the keywords
            keyword_tokens = self.preprocess_text(self.keywords[self.dataframe.index.tolist().index(i)])
            # For each video
            for j in file.dataframe.index.tolist():
                keyword_scores[i][j] = {"score": 0}
                total_words = len(file.dataframe['text'][j])
                index = file.dataframe.index.tolist().index(j)
                word_tokens = self.preprocess_text(file.text[index])
                for token in word_tokens:
                    if token in keyword_tokens:
                        keyword_scores[i][j]["score"] += 1 / total_words
        # Sort the files.id for each self.id
        for i in keyword_scores:
            keyword_scores[i] = dict(sorted(keyword_scores[i].items(), key=lambda x: x[1]["score"], reverse=True))
        return keyword_scores


    def best_match(self, file):
        """
            Returns the best two videos for each article
        """
        keyword_scores = self.score_keywords_mix(file)
        for i in keyword_scores:
            keyword_scores[i] = {k: keyword_scores[i][k] for k in list(keyword_scores[i])[:2]}
        self.write_json(keyword_scores)
        return keyword_scores
    
    def write_json(self, scores_dic):
        """
            Writes given dictionary to a file in a more readable format
        """
        with open('entrega.json', "w+") as file:
            file.write(json.dumps(scores_dic, indent=4, sort_keys=True))



    # # Obtener scores de keywords en un texto (Ana Version)
    # def score_keywords_ana(self, file):
    #     # For each article
    #     dic = {}
    #     for i in self.dataframe.index.tolist():
    #         dic[i] = {}
    #         # For each video
    #         for j in file.dataframe.index.tolist():
    #             dic[i] = {j:{}}
    #             # For each keyword of article i
    #             for k in self.dataframe['keywords'][i]:
    #                 # Score function
    #                 dic[i][j][k] = 0
    #     print(dic)

    # # Obtener scores de keywords en un texto (Ana Version)

