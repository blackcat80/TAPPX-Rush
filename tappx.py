import pandas as pd
import json
import numpy as np
#import nltk
from numpy.linalg import norm
import string
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake
import re
import json

class File:

    def __init__(self, file):
        # Create dataframe using panda
        self.dataframe = pd.read_json(f"{file}.json").transpose()

        # Create lists of variables for usability
        self.id       = self.dataframe.index
        self.keywords = self.dataframe["keywords"]
        self.text     = self.dataframe["text"].apply(self.clean_single_text)

        # Save model to Class 
        # https://huggingface.co/hiiamsid/sentence_similarity_spanish_es
        self.model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

        r = Rake()
        # Create new column of dataframe with a sentence created by its keywords
        self.keywords_str = []
        for i in self.id:
            r.extract_keywords_from_text(self.text[i])
            filtered_keys=[]
            for phrase in r.get_ranked_phrases_with_scores():
                if phrase[0] > 80:
                    filtered_keys.append(phrase[1])
            if ('title' in self.dataframe.columns):
                filtered_keys.append(self.clean_single_text(self.dataframe["title"][i]))
            self.keywords_str.append(filtered_keys + self.keywords[i])
        self.dataframe['keywords_mix'] = self.keywords_str

    def clean_single_text(self, text):
        """
            Usage of regular expressions to detect errors in the text, and replace them with the corresponding correction.
        """
        text = text.strip()
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\{.*?\}|\[.*?\]|\(.*?\))', '', text)
        text = re.sub(r'\n|\r', ' ', text)
        text = re.sub(r'\d', '', text)
        text = ''.join(c for c in text if c not in string.punctuation)
        text = re.sub(r'\s+', ' ', text)
        return text.lower()

    def cosin_model(self, first, sec):
        """
            Returns Cosine Similarity of the two given sentences by using an AI model 
        """
        sentences = [first, sec]
        embeddings = self.model.encode(sentences)
        A = np.array(embeddings[0])
        B = np.array(embeddings[1])
        return np.dot(A,B)/(norm(A)*norm(B))
        
    def score_file(self, file):
        """
            Creates a dictionary for self.id and each file.id score
        """
        # For each article
        dic = {}
        for i in self.id:
            dic[i] = {}
            # For each video
            for j in file.id:
                # Score function
                ret = self.cosin_model(self.dataframe['keywords_mix'][i], file.dataframe['keywords_mix'][j])
                ret *= 100
                dic[i][j] = {"score": ret}
        return dic
        
    def write_json(self, scores_dic):
        """
            Writes given dictionary to a file in a more readable format
        """
        with open('entrega.json', "w+") as file:
            file.write(json.dumps(scores_dic, indent=4, sort_keys=True))

    def best_match(self, file):
        """
            Returns the best two videos for each article
        """
        keyword_scores = self.score_file(file)
        # Sort the files.id for each self.id
        for i in keyword_scores:
            keyword_scores[i] = dict(sorted(keyword_scores[i].items(), key=lambda x: x[1]["score"], reverse=True))
        # Filter for 2 largest scores per self.id
        for i in keyword_scores:
            keyword_scores[i] = {k: keyword_scores[i][k] for k in list(keyword_scores[i])[:2]}
        # Create entrega.json file
        self.write_json(keyword_scores)
        return keyword_scores

if __name__ == "__main__":
    articles = File("articles")
    videos   = File("videos")
    
    articles.best_match(videos)
