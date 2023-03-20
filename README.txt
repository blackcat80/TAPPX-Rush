OBJETIVO RETO:

El objetivo es asignar un conjunto de vídeos de los que se encuentran en videos.json a cada artículo de los que se encuentra en articulos.json. 
Los vídeos que se asignan a cada artículo deben tener relación de contexto con este.

• El reto tiene que entregarse en un archivo comprimido con los siguientes elementos:
    ◦ Todo el código y dependencias.
    ◦ Archivo de requerimientos.
    ◦ Un archivo json con las características mencionadas en el apartado entrega.
    ◦ Un archivo README.txt con la explicación del funcionamiento del programa, la definición del score y cualquier otra información que se considere relevante.


INSTRUCCIONES - Estructura del proyecto

1. Crear archivo requirements.txt (a traves de 'pip install -r requirements.txt', este se encargara de instalar las dependencias usadas en el programa)

2. Crear archivo README.txt (en el explicaremos el objetivo y funcionamiento del programa, la definicion del score y toda información relevante)

3. Creamos un archivo tappx.py (en el que ejecutaremos nuestro código)

4. Extraer los datos de los archivos JSON (en concreto las id de los articulos, las keywords y los textos asociados), y guardar los textos en una lista

5. Limpiar los textos extraídos, gramatica y semánticamente. 

6. Buscar las keywords y sus scores. (Podemos usar una librería de PLN (procesamiento de lenguaje natural), como 'spacy' para procesar los textos
   y extraer sus keywords y  sus scores)

7. Enlazar los artículos con los vídeos: (mínimo 2 vídeos por articulo). Podemos iterar sobre la lista de diccionarios de vídeos y comparar los keywords de cada vídeo con los keywords de cada
   artículo para encontrar la mejor coincidencia. (quizás podemos hacer un .join de las keywords generadas por nosotros con las dadas en los archivos json)

8. Por último crear un arhivo entrega.json con dichas asignaciones con el siguiente formato:

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