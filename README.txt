README:

This project aims to assign a set of videos to each article based on its context. For this, a json file of articles and another json file of videos are used, which contain relevant information about each one of them.

To run the program, the tappx.py file must be compiled with Python 3 using the "python3 tappx.py" command. Once the process is finished, a entrega.json file will be generated with the matching results.

The entrega.json file has the following format:
    {
    "id-del-articulo1": {
        "id_del_video_relacionado1": {
            "score":"score-de-la-relación"
        },
        "id_del_video_relacionado2":{
            "score":"score-de-la-relación"
        },
        .
        .            
    },
    "id-del-articulo2": {
        "id_del_video_relacionado1": {
            "score":"score-de-la-relacion"
        },
        .   
        .            
    }
    .
    .        
}

The logic used to calculate the score is based on the vectorial similarity of the cosine ('cosine similarity') between the keywords of the article and the keywords of the video. For this, two pre-trained models are used: BERT for the creation of vectors and Rake for obtaining new keywords. In addition, the title of the article is considered in the comparison.

The score has a range between 0 and 100, with 100 being the closest relationship between the article and the video.

The code is organized in a File class, which receives the name of the json file as a parameter and creates a dataframe from the data. Also, new keywords are generated using the Rake algorithm and added to the dataframe.

For the processing of the json files, two objects of the File class are created: 'articles' and 'videos'. Then, the 'best_match' function of the 'articles' object is called to generate the delivery.json file with the article and video pairing.

Dependencies are registered by pip freeze and the output is redirected to the requirements.txt file for easy installation.

It is important to mention that due to the execution time of the program, the entrega.json file is provided with the results already obtained.

Project members: Ana R., Nil B. and Christian S.
