README:

Este proyecto tiene como objetivo asignar un conjunto de vídeos a cada artículo en función de su contexto. Para esto, se utiliza un archivo json de artículos y otro archivo json de vídeos, los cuales contienen información relevante acerca de cada uno de ellos.

Para ejecutar el programa, se debe compilar el archivo tappx.py con Python 3 mediante el comando "python3 tappx.py". Una vez finalizado el proceso, se generará un archivo entrega.json con los resultados del emparejamiento.

El archivo entrega.json tiene el siguiente formato:
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

La lógica utilizada para calcular el score se basa en la similitud vectorial del coseno ('cosine similarity') entre las keywords del artículo y las keywords del video. Para ello, se utilizan dos modelos pre-entrenados: BERT para la creación de vectores y Rake para la obtención de nuevas keywords. Además, se considera el título del artículo en la comparación.

El score tiene un rango entre 0 y 100, siendo 100 la relación más cercana entre el artículo y el video.

El código se encuentra organizado en una clase File, la cual recibe como parámetro el nombre del archivo json y crea un dataframe a partir de los datos. Además, se generan nuevas keywords utilizando el algoritmo Rake y se añaden al dataframe.

Para el procesamiento de los archivos json, se crean dos objetos de la clase File: 'articles' y 'videos'. Luego, se llama a la función 'best_match' del objeto 'articles' para generar el archivo entrega.json con el emparejamiento de artículos y videos.

Se registra las dependencias por pip freeze y se redirecciona el output al archivo requirements.txt para una fácil instalación de las mismas.

Es importante mencionar que debido al tiempo de ejecución del programa, se facilita el archivo entrega.json con los resultados ya obtenidos.

Miembros del proyecto: Ana R., Nil B. y Christian S.
