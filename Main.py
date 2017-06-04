# -*- coding: utf-8 -*-
"""
Alejandra Perez
Python 3
"""

import time
#Import numpy
import numpy as np
#Import unicode para el manejo de utf8
import unicodedata
#Import csv (lectura de archivos csv)
import csv
#Import nltk (Natural Language)
#import nltk
#nltk.download()
from nltk.tokenize import word_tokenize
#Import regex
import re
#Importar un aleatorio
import random
import math
random.seed = 0
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
        #Check
        peso= 40*(contador / len(tweet.split(sep=' '))) - 20.0
        lista_categoria.append(peso)
        contador = 0
    matriz.append(lista_categoria)
    
#funcion para limpiar el texto                
def limpiar_texto(texto):
    texto = texto.lower()
    texto = eliminar_tildes(texto)
    texto = eliminar_puntuacion(texto)
    return texto

#funcion que se encarga de cargar la lista de stop words en español
def cargar_stop_words(lista):
    from nltk.corpus import stopwords # Import the stop word list
    listaTemporal = stopwords.words("spanish")
    for palabra in listaTemporal:
        lista.append(eliminar_tildes(palabra))

#Class variables
class RedNeuronal:
    def __init__(self,numEntradas,numOcultas,numSalidas):
        self.numEntradas = numEntradas
        self.numOcultas = numOcultas
        self.numSalidas = numSalidas
        self.entradas = [0.0] * numEntradas
        self.EOPesos = self.hacer_matriz(numEntradas,numOcultas,0.0)
        self.OBiases = [0.0] * numOcultas
        self.OSalidas = [0.0] * numOcultas
        self.OSPesos = self.hacer_matriz(numOcultas,numSalidas,0.0)
        self.SBiases = [0.0] * numSalidas
        self.salidas = [0.0] * numSalidas
        self.inicializar_pesos()
        
    def hacer_matriz(self,filas, columnas, valor):
        resultado = [None]* filas
        for i in range(0,filas):
            resultado[i] = [None]*columnas
        for j in range(0,filas):
            for k in range(0,columnas):
                resultado[j][k]=valor
        return resultado
    def inicializar_pesos(self):
        numPesos = (self.numEntradas*self.numOcultas) + (self.numOcultas * self.numSalidas) + self.numOcultas + self.numSalidas
        pesosIniciales = []
        for i in range(0,numPesos):
            pesosIniciales.append((0.001 - 0.0001)*random.random()+0.0001)
        self.set_pesos(pesosIniciales)
        
    def set_pesos(self,pesos):
        numPesos = (self.numEntradas*self.numOcultas) + (self.numOcultas * self.numSalidas) + self.numOcultas + self.numSalidas
        #Podria omitirse
        if len(pesos) != numPesos:
            print(str(len(pesos)) + " " + str(numPesos))
            raise Exception("Fuck the police")
        k = 0
        for i in range(0,self.numEntradas):
            for j in range(0,self.numOcultas):
                #k = k+1
                self.EOPesos[i][j] = pesos[k+1]
        for i in range(0,self.numOcultas):
            #k = k+1
            self.OBiases[i] = pesos[k+1]
        for i in range(0,self.numOcultas):
            for j in range(0,self.numSalidas):
                #k = k+1
                self.OSPesos[i][j] = pesos[k+1]
        for i in range(0,self.numSalidas):
            #k = k + 1
            #print(k)
            #print(i)
            self.SBiases[i] = pesos[k+1]
            
    def get_pesos(self):
        numPesos =  (self.numEntradas*self.numOcultas) + (self.numOcultas * self.numSalidas) + self.numOcultas + self.numSalidas
        resultado = [numPesos]
        k = 0
        for i in range(0,len(self.EOPesos)):
            for j in range(0,len(self.EOPesos[0])):
                k = k+1
                resultado[k] = self.EOPesos[i][j]
        for i in range(0,len(self.OBiases)):
            k = k+1
            resultado[k]= self.OBiases[i]
        for i in range(0,len(self.OSPesos)):
            for j in range(0,len(self.OSPesos[0])):
                k = k+1
                resultado[k]=self.OSPesos[i][j]
        for i in range(0,self.SBiases):
            k = k + 1
            resultado[k] = self.SBiases[i]
        return resultado
    
    def computar_salida(self,xvalores):
        hSums = [0.0]*self.numOcultas
        oSums = [0.0] * self.numSalidas
        self.entradas = list(xvalores)
        for j in range(0,self.numOcultas):
            for i in range(0,self.numEntradas):
                hSums[j] += self.entradas[i]*self.EOPesos[i][j]
        for i in range(0,self.numOcultas):
            hSums[i] =  hSums[i] + self.OBiases[i]
        for i in range(0,self.numOcultas):
            self.OSalidas[i] = self.hyper_tan(hSums[i])
        for j in range(0,self.numSalidas):
            for i in range(0,self.numOcultas):
                oSums[j] += self.OSalidas[i] * self.OSPesos[i][j]
        for i in range(0, self.numSalidas):
            oSums[i] += self.SBiases[i]
        softOut = self.softmax(oSums)
        self.salidas = list(softOut)
        retResolt = list(self.salidas)
        return retResolt
    def hyper_tan(self,x):
        if x < -20:
            return -1.0
        elif x > 20:
            return 1.0
        else:
            return math.tanh(x)
    def softmax(self,oSums):
        suma = 0.0
        for i in range(0,len(oSums)):
            suma += math.exp(oSums[1])
        result = [None]* len(oSums)
        for i in range(0,len(oSums)):
            print(oSums)
            result[i] = math.exp(round(oSums[i],1)) / suma
        return result
    def train(self,trainData,maxEpochs,learnRate,momentum):
        hoGrads = self.hacer_matriz(self.numOcultas,self.numSalidas,0.0)
        obGrads = [None]* self.numSalidas
        ihGrads = self.hacer_matriz(self.numEntradas,self.numOcultas,0.0)
        hbGrads = [None] * self.numOcultas
        oSignals = [None] * self.numSalidas
        hSignals = [None] * self.numOcultas
        ihPrevWeightsDelta = self.hacer_matriz(self.numEntradas,self.numOcultas,0.0)
        hPrevBiaseseDelta = [0.0] * self.numOcultas
        hoPrevWeightsDelta = self.hacer_matriz(self.numOcultas,self.numSalidas,0.0)
        oPrevBiasesDelta = [0.0] * self.numSalidas
        epoch = 0
        xValues = [None] * self.numEntradas
        tValues = [None]* self.numSalidas
        derivative = 0.0
        errorSignal = 0.0
        sequence = [None]* len(trainData)
        for i in range(0,len(sequence)):
            sequence[i] = i
        errInterval = maxEpochs / 10
        while epoch < maxEpochs:
            epoch += 1
            if epoch % errInterval == 0 and epoch < maxEpochs:
                trainErr = self.Error(trainData)
                print("epoch = "+epoch + " error = "+trainErr)
            self.Shuffle(sequence)
            for ii in range(0,len(trainData)):
                idx = sequence[ii]
                xValues = list(trainData[idx])
                tValues = list(trainData[idx][self.numEntradas-2:])
            
                self.computar_salida(xValues)
                for k in range(0,self.numSalidas):
                    errorSignal = tValues[k] - self.salidas[k]
                    derivative = (1 - self.salidas[k] * self.salidas[k])
                    oSignals[k] = errorSignal * derivative
                for j in range(0,self.numOcultas):
                    for k in range(0,self.numSalidas):
                        hoGrads[j][k] = oSignals[k]*self.OSalidas[j]
                for k in range(0,self.numSalidas):
                    obGrads[k] = oSignals[k] * 1.0
                    
                for j in range(0,self.numOcultas):
                    derivative = (1+self.OSalidas[j])*(1-self.OSalidas[j])
                    suma = 0.0
                    for k in range(self.numSalidas):
                        suma += oSignals[k] * self.OSPesos[j][k]
                    hSignals[j] = derivative * suma
                for i in range(0,self.numEntradas):
                    for j in range(0,self.numOcultas):
                        ihGrads[i][j] = hSignals[j] * self.entradas[i]
                for j in range(0,self.numOcultas):
                    hbGrads[j] = hSignals[j] * 1.0
                for i in range(0,self.numEntradas):
                    for j in range(0,self.numOcultas):
                        delta = ihGrads[i][j] * learnRate
                        self.EOPesos[i][j] += delta 
                        self.EOPesos[i][j] += ihPrevWeightsDelta[i][j]*momentum
                        ihPrevWeightsDelta[i][j] = delta
                for j in range(0,self.numOcultas):
                    delta = hbGrads[j] * learnRate
                    self.OBiases[j] += delta
                    self.OBiases[j] += hPrevBiaseseDelta[j]*momentum
                    hPrevBiaseseDelta[j] = delta
                for j in range(0,self.numOcultas):
                    for k in range(0,self.numSalidas):
                        delta = hoGrads[j][k] * learnRate
                        self.OSPesos[j][k] += delta
                        self.OSPesos[j][k] += hoPrevWeightsDelta[j][k] * momentum
                        hoPrevWeightsDelta[j][k] = delta
                for k in range(0,self.numSalidas):
                    delta = obGrads[k] * learnRate
                    self.SBiases[k] += delta
                    self.SBiases[k] += oPrevBiasesDelta[k] * momentum
                    oPrevBiasesDelta[k] = delta
        bestWts = self.get_pesos()
        return bestWts
    
    def Shuffle(self,sequence):
        for i in range(0,len(sequence)):
            r = random.randrange(i,len(sequence))
            tmp = sequence[r]
            sequence[r] = sequence[i]
            sequence[i] = tmp
    def Error(self,trainData):
        sumSquaredError = 0.0
        xValues = [self.numEntradas]
        tValues = [self.numSalidas]
        for i in range(0,len(trainData)):
            xValues = list(trainData[i])
            tValues = list(trainData[i][self.numEntradas:])
            yValues = self.computar_salida(xValues)
            for j in range(0,self.numSalidas):
                err = tValues[j] - yValues[j]
                sumSquaredError += err * err
        return sumSquaredError / len(trainData)
    def Accuracy(self,testData):
        numCorrect = 0
        numWrong = 0
        xValues = [self.numEntradas]
        tValues = [self.numSalidas]
        yValues = []
        for i in range(0,len(testData)):
            xValues = list(testData[i])
            tValues = list(testData[i][self.numEntradas:])
            yValues = self.computar_salida(xValues)
            maxIndex = self.MaxIndex(yValues)
            tMaxIndex= self.MaxIndex(tValues)
            if maxIndex == tMaxIndex:
                numCorrect += 1
            else:
                numWrong += 0
        return (numCorrect*1.0) / (numWrong+numCorrect)
    def MaxIndex(self,vector):
        granIndice = 0
        mayorValor = vector[0]
        for i in range(0,len(vector)):
            if vector[i] > mayorValor:
                mayorValor = vector[i]
                granIndice = i
        return granIndice

def split_data(datos_origen,porcentaje,datos_entrenamiento,datos_pruebas):
    totFilas = len(datos_origen)
    numTrainFilas = int((totFilas*porcentaje))
    #print(numTrainFilas)
    numTestFilas = totFilas - numTrainFilas
    copia = [None]* len(datos_origen)
    for i in range(0,len(copia)):
        copia[i] = datos_origen[i]
    for i in range(0,len(copia)):
        r = random.randrange(i,len(copia))
        tmp = copia[r]
        copia[r] = copia[i]
        copia[i] = tmp
    for i in range(0,numTrainFilas):
        #print(i)
        datos_entrenamiento.append(copia[i])
    for i in range(0,numTestFilas):
        datos_pruebas.append(copia[i+numTrainFilas])
        #Main del programa
def preparar_matriz(lista_datos,lista_labels):
    matriz1=[]
    fila = []
    for i in range(0,len(lista_datos[0])):
        for j in range(0,len(lista_datos)):
            fila.append(lista_datos[j][i])
        if lista_labels[i] == 1:
            fila.append(0)
            fila.append(1)
        else:
            fila.append(1)
            fila.append(0)
        #print(fila)
        matriz1.append(list(fila))
        fila.clear()
    return matriz1
    
if __name__ == "__main__":
    cargar_stop_words(lista_stop)
    leerFicheroALista('tweets.csv',listaEntrada)
    leerFicheroALista('hashtags.csv',listaHashtags)
    for archivo in lista_archivos:
        clasificar(matriz_clasificaciones,listaEntrada,archivo+'.csv',archivo)
    matriz1 =preparar_matriz(matriz_clasificaciones,lista_label)
    del matriz_clasificaciones
    del lista_label
    del listaEntrada
    del listaHashtags
    #print(matriz1)
    #print(listaEntrada)
    del lista_stop
    #print(matriz1)
    #Neural
    numero_features = len(matriz1[0])
    lista_train = []
    lista_test = []
    split_data(matriz1,0.7,lista_train,lista_test)
    #print(len(lista_test))
    numNodosHidden=5
    numSalidas=2
    semilla=1
    red = RedNeuronal(numero_features,numNodosHidden,numSalidas)
    epocasMaximas = 1000
    learnRate = 0.05
    momentun = 0.01
    pesos = red.train(matriz1,epocasMaximas,learnRate,momentun)
    print(pesos)    