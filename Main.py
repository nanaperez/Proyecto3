# -*- coding: utf-8 -*-
"""
Alejandra Perez
Python 3
"""
#Import numpy
import numpy as np
#Import unicode para el manejo de utf8
import unicodedata
#Import csv (lectura de archivos csv)
import csv
#Import nltk (Natural Language)
import nltk
#nltk.download()
from nltk.tokenize import word_tokenize
#Import regex
import re
#Import string
import string
#Import math
import math

## Globales
listaEntrada = []
lista_label = []
listaHashtags = []
lista_stop = []
lista_archivos=['arquitectura','arte','ciudades', 'danza', 'escultura', 
'festivales', 'fotografia','galerias', 'gastronomia', 'graffitis', 
'literatura', 'museo', 'musica','pintura', 'poesia', 'teatro','Hashtags']
# Arquitectura, arte, ciudades, danza escultura, festivales, fotografia,galerias
#, gastronomia, graffitis, literatura, museo, musica,pintura, poesia, teatro,
#Hashtags
matriz_clasificaciones = []
## Metodos
#funcion que elimina las tildes de las palabras
def eliminar_tildes(s):
   return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))

#funcion que elimina puntuacion con regex
def eliminar_puntuacion(s):
    #regex = re.compile('[%s]' % re.escape(string.punctuation))
    return re.sub(u"[^a-zA-Z0-9]"," ", s)

##Si falla el codigo utf, usar ISO-8859-1
#funcion que lee un fichero y lo convierte en lista  
def leerFicheroALista(rutaArchivo,listaDestino):
    with open(rutaArchivo,'r',encoding="ISO-8859-1") as archivo:
        csv.Dialect.skipinitialspace= True
        csvreader = csv.reader(archivo, delimiter = ';')
        for row in csvreader:
            listaDestino.append(limpiar_texto(row[0]))
            if len(row) >= 2:
                lista_label.append(limpiar_texto(row[1]))

#Funcion para una clasificacion a fuerza bruta
def clasificar(matriz,lista_tweets,url_archivo,categoria):
    conjunto= set()
    lista_categoria=[]
    lista_categoria.append(categoria)
    contador = 0
    with open(url_archivo,'r',encoding="ISO-8859-1") as archivo:
        csvreader = csv.reader(archivo, delimiter = ';')
        for row in csvreader:
            conjunto.add(row[0])
    for tweet in lista_tweets:
        for palabra in conjunto:
            if palabra in tweet:
                contador = contador + 1
        lista_categoria.append((contador / len(tweet.split(sep=' '))))
        contador = 0
    matriz.append(lista_categoria)
    
#funcion para limpiar el texto                
def limpiar_texto(texto):
    texto = texto.lower()
    texto = eliminar_tildes(texto)
    texto = eliminar_puntuacion(texto)
#    listica = texto.split(sep=' ')
#    tweet = []
#    for palabra in listica:
#        if palabra not in lista_stop and palabra is not '':
#            tweet.append(palabra)
    return texto

#funcion que se encarga de cargar la lista de stop words en español
def cargar_stop_words(lista):
    from nltk.corpus import stopwords # Import the stop word list
    listaTemporal = stopwords.words("spanish")
    for palabra in listaTemporal:
        lista.append(eliminar_tildes(palabra))
 
#funcion que separa las instancias con clasificacion invalida    
"""
def clasificaciones_invalidas(instancias, clases_invalidas=["null"], ind_clase=0):

    Separamos las instancias con una clasificacion valida de las que no la tienen.
    :param instancias: Instancias a separar.
    :param ind_clase: Indice de la instancia que contiene su clasificacion.
    :param clases_invalidas: Valores de clasificacion invalidos.
    
    invalidas = []
    for c in instancias:
        if (c[ind_clase] in clases_invalidas):
            instancias.append(c)
            instancias.remove(c)
    return invalidas, instancias

#funcion que extrae las caracteristicas de un texto
def rasgos_del_texto(texto, listado_palabras):
    if(not texto):
        print("Texto nulo")
        return None
    palabras_del_texto = set(texto)
    caracteristicas = {}
    for p in listado_palabras:
        caracteristicas['contiene({})'.format(p)] = (p in palabras_del_texto)
    return caracteristicas

#funcion que retorna un listado de las palabras comunes
def listado_palabras(instancias, ind_texto=1, ind_clase=0, invalidos=["null"]):
    conjunto_palabras = clas.cadena_de_palabras(instancias, [2])
    conjunto_palabras = nltk.word_tokenize(conjunto_palabras)
    
    #print conjunto_palabras
    
    listado_palabras = nltk.FreqDist(w.lower() for w in conjunto_palabras)
    
    return listado_palabras

#funcion que clasifica una instancia en base a las palabras comunes que contiene
def clasificacion_binaria_palabras(dist_frec, instancias, n, clases_invalidas):
    
    :param distr_frec: Ruta del CSV con las instancias a clasificar.
    :param n: Numero de palabras mas comunes a tener en cuenta.
    :param clases_invalidas: Las instancias que las tengan no estan clasificadas.
    
    #escogemos las N palabras mas comunes
    mas_comunes = list(dist_frec)[:n]
    
    invalidas, validas = clas.clasificaciones_invalidas(instancias, clases_invalidas)
    #A partir de un texto, creamos un hash con las caracteristicas de dicho texto.
    inst_validas = [(clas.rasgos_del_texto(d, mas_comunes), c) for (c, t, d) in validas]
    inst_invalidas = [(clas.rasgos_del_texto(d, mas_comunes), c) for (c, t, d) in invalidas]
    half = math.floor(len(instancias) / 2)
    train_set , test_set = inst_validas [:half], inst_invalidas[half:]
    # train_set , test_set = inst_validas , inst_invalidas
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    resultado = {}
    resultado['precision'] = nltk.classify.accuracy(classifier, test_set)
    return resultado
"""


#funcion bag of words
def bag_of_words():
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    #data
    bag_of_words = vectorizer.fit(data)
    bag_of_words = vectorizer.transform(data)
    print (bag_of_words)
    print (vectorizer.vocabulary_get("cultura"))
    
#funcion donde se utiliza Naive_Bayes classificator
def Naive(listaInicial):
    return
 
if __name__ == "__main__":
    cargar_stop_words(lista_stop)
    leerFicheroALista('tweets.csv',listaEntrada)
    leerFicheroALista('hashtags.csv',listaHashtags)
#    print(listaHashtags)
#    print(len(listaEntrada))
    for archivo in lista_archivos:
        clasificar(matriz_clasificaciones,listaEntrada,archivo+'.csv',archivo)
    print(matriz_clasificaciones)