# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import numpy as np
import unicodedata
import csv
import nltk
#nltk.download()
from nltk.tokenize import word_tokenize
# Import regex
import re
# Import string
import string

## Globales
listaEntrada= []
lista_label=[]
listaHashtags = []
lista_stop = []

## Metodos
def eliminar_tildes(s):
   return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
   
def eliminar_puntuacion(s):
    #regex = re.compile('[%s]' % re.escape(string.punctuation))
    return re.sub(u"[^a-zA-Z]"," ", s)
    
def leerFicheroALista(rutaArchivo,listaDestino):
    with open(rutaArchivo,'r',encoding="utf8") as archivo:
        csv.Dialect.skipinitialspace= True
        csvreader = csv.reader(archivo, delimiter = ';')
        for row in csvreader:
            listaDestino.append(limpiar_texto(row[0]))
            if len(row) >= 2:
                lista_label.append(limpiar_texto(row[1]))
                
def limpiar_texto(texto):
    texto = texto.lower()
    texto = eliminar_tildes(texto)
    texto = eliminar_puntuacion(texto)
    listica = texto.split(sep=' ')
    tweet = []
    for palabra in listica:
        
        if palabra not in lista_stop and palabra is not '':
            tweet.append(palabra)
    return tweet     
   
def Naive(listaInicial):
    return

def cargar_stop_words(lista):
    from nltk.corpus import stopwords # Import the stop word list
    listaTemporal = stopwords.words("spanish")
    for palabra in listaTemporal:
        lista.append(eliminar_tildes(palabra))

if __name__ == "__main__":
    cargar_stop_words(lista_stop)
    leerFicheroALista('tweets.csv',listaEntrada)
    leerFicheroALista('hashtags.csv',listaHashtags)
    print(listaEntrada)
    print(len(listaEntrada))
    