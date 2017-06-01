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

## Globales
listaEntrada = []
lista_label = []
listaHashtags = []
lista_stop = []
lista_archivos=['arquitectura','arte','ciudades', 'danza', 'escultura', 
'festivales', 'fotografia','galerias', 'gastronomia', 'graffitis', 
'literatura', 'museo', 'musica','pintura', 'poesia', 'teatro','hashtags']
# Arquitectura, arte, ciudades, danza escultura, festivales, fotografia,
# galerias, gastronomia, graffitis, literatura, museo, musica,pintura, 
# poesia, teatro, hashtags
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
        csvreader = csv.reader(archivo, delimiter = ';')
        for row in csvreader:
            listaDestino.append(limpiar_texto(row[0]))
            if len(row) >= 2:
                opcion = limpiar_texto(row[1])
                if opcion == 'seleccionado':
                    lista_label.append(1)
                else:
                    lista_label.append(0)
                #lista_label.append(limpiar_texto(row[1]))

#Funcion para una clasificacion a fuerza bruta
def clasificar(matriz,lista_tweets,url_archivo,categoria):
    conjunto= set()
    lista_categoria=[]
    #lista_categoria.append(categoria)
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
        
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def red_magica(datos,respuestas):
    #Input dataset
    x = np.array(datos,dtype='float64').T
    #Answers
    y = np.array(respuestas,dtype='float64')
    np.random.seed(1)
    #w1
    syn0 = 2*np.random.random((17,4)) - 1
    #w2
    syn1 = 2*np.random.random((4,794)) - 1
    entregar = {}
    for j in range(600):
        #Input layer
        l0 = x
        #Hidden layer
        l1 = nonlin(np.dot(l0,syn0))
        #Output layer
        l2 = nonlin(np.dot(l1,syn1))
        l2_error = y - l2
        if (j% 100) == 0:
            print("Error:" + str(np.mean(np.abs(l2_error))))
            #print(l2)
        l2_delta = l2_error*nonlin(l2,deriv=True)
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * nonlin(l1,deriv=True)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)
        entregar["w1"] = syn0
        entregar["w2"] = syn1
        entregar["b1"] = l1
        entregar["b2"] = l2
    return entregar

def predict(model, x):
    W1, b1, W2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

#Main del programa
if __name__ == "__main__":
    cargar_stop_words(lista_stop)
    leerFicheroALista('tweets.csv',listaEntrada)
    leerFicheroALista('hashtags.csv',listaHashtags)
    for archivo in lista_archivos:
        clasificar(matriz_clasificaciones,listaEntrada,archivo+'.csv',archivo)
    lista_label.remove(0)
    #print(matriz_clasificaciones)
    #print(listaEntrada)
    del lista_stop
    modelo = red_magica(matriz_clasificaciones,lista_label)
    print(modelo)
    ejemplo = [0.6,]
    print(predict(modelo,np.random.random((17,1))))
