import os.path
import sys
import numpy as np
import pandas as pd
import csv
import time
from sklearn import tree
from sklearn import svm
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # doctest: +SKIP
import pickle
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from glob import glob






#INICIO DEL PROGRAMA

CLASIFICADORES = np.array(['DT','NB','ANN','SVM','RF','GBM','VC'])
METRICAS = np.array(['Fscore','Precision','Sensibilidad','Total'])
NUM_CARACT = [5,10,15,20,25]

#LECTURA DE ARGUMENTOS
num_caract = sys.argv[1]
selector = sys.argv[2]
print("ARGUMENTOS: num_caract=" + str(num_caract) + " selector=" + str(selector))




#APERTURA FICHERO
print ("LECTURA DE FICHEROS")
le = pickle.load(open('./modelos/le.sav', 'rb'))

CLASES = le.classes_

compararPCA = np.array([])
for num in NUM_CARACT:
    filepath = "./resultados/features" + str(num) + "selectorPCA/bacc_FeaturesSelector.dat"
    with open(filepath) as fp:
       linea = fp.readline()
       suma = float(linea)
       cnt = 1
       while linea:
           linea = fp.readline()
           if linea == '':
               break
           suma = suma + float(linea)
           cnt = cnt + 1
    media = suma/cnt
    compararPCA = np.append(compararPCA, media)

compararRFE = np.array([])
for num in NUM_CARACT:
    filepath = "./resultados/features" + str(num) + "selectorRFE/bacc_FeaturesSelector.dat"
    with open(filepath) as fp:
       linea = fp.readline()
       suma = float(linea)
       cnt = 1
       while linea:
           linea = fp.readline()
           if linea == '':
               break
           suma = suma + float(linea)
           cnt = cnt + 1
    media = suma/cnt
    compararRFE = np.append(compararRFE, media)



#Crear array con 'balanced_accuracy_score' de cada clasificador

baccs_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/features" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-bacc.dat"
    with open(file_path,"r") as f:
        num = np.array([float(f.readline())])
        baccs_t = np.append(baccs_t, num)

#Crear array con 'precision' de cada clasificador

p_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/features" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-precision.dat"
    with open(file_path,"r") as f:
        num = np.array([float(f.readline())])
        p_t = np.append(p_t, num)


#Crear array con 'Tiempo_entrenamiento' de cada clasificador

t_entrenamiento_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/features" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-Tiempo_entrenamiento.dat"
    with open(file_path,"r") as f:
        num = np.array([float(f.readline())])
        t_entrenamiento_t = np.append(t_entrenamiento_t, num)


#Crear array con 'Tiempo_clasificacion' de cada clasificador

t_clasificacion_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/features" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-Tiempo_clasificacion.dat"
    with open(file_path,"r") as f:
        num = np.array([float(f.readline())])
        t_clasificacion_t = np.append(t_clasificacion_t, num)





#CREACIÓN DE TABLAS
print ("CREACIÓN DE TABLAS")
#Crear tabla por cada clasificador mostrando las diferentes métricas
for clf in CLASIFICADORES:
    matriz_metricas = np.empty(shape=(len(METRICAS), len(le.classes_) ))
    for i in range(0,len(METRICAS)):
        file_path = "./resultados/features" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-" + METRICAS[i] + ".npy"
        v_metricas = np.load(file_path)
        matriz_metricas[i] = v_metricas.round(decimals=3)

    fig = go.Figure(data=[go.Table(header=dict(values=np.append(['Clase\Metrica'],METRICAS)),
                     cells=dict(  values=np.transpose(np.concatenate((le.classes_[:,None],np.transpose(matriz_metricas)),axis=1))  ))
                         ])
    fig.write_image("./graficos/features"  + str(num_caract) + "selector" + str(selector) + "/tab" + clf + "-Metricas.png")






#CREACIÓN DE GRÁFICAS
print ("CREACIÓN DE GRÁFICAS")

#GRAFICAS DE COLUMNAS
figBacc_t = go.Figure([go.Bar(x=CLASIFICADORES, y=baccs_t)])
figBacc_t.update_layout(
    xaxis_title="Clasificadores",
    yaxis_title="Exactitud balanceada"
)
figBacc_t.write_image("./graficos/features"  + str(num_caract) + "selector" + str(selector) + "/figBacc_t.png")


figP_t = go.Figure([go.Bar(x=CLASIFICADORES, y=p_t)])
figP_t.update_layout(
    xaxis_title="Clasificadores",
    yaxis_title="Precisión"
)
figP_t.write_image("./graficos/features"  + str(num_caract) + "selector" + str(selector) + "/figP_t.png")

figTRT_t  = go.Figure([go.Bar(x=CLASIFICADORES, y=(t_entrenamiento_t/60))])
figTRT_t.update_layout(
    xaxis_title="Clasificadores",
    yaxis_title="Tiempo de entrenamiento (min)"
)
figTRT_t.write_image("./graficos/features"  + str(num_caract) + "selector" + str(selector) + "/figTRT_t.png")

figTST_t = go.Figure([go.Bar(x=CLASIFICADORES, y=t_clasificacion_t)])
figTST_t.update_layout(
    xaxis_title="Clasificadores",
    yaxis_title="Tiempo de clasificación (s)"
)
figTST_t.write_image("./graficos/features"  + str(num_caract) + "selector" + str(selector) + "/figTST_t.png")





figCompararNumCaract = go.Figure([go.Scatter()])
figCompararNumCaract.update_layout(
    xaxis_title="Clasificadores",
    yaxis_title="E"
)
figCompararNumCaract.add_trace(go.Scatter(x=NUM_CARACT, y=compararRFE,
                    name='RFE'))
figCompararNumCaract.add_trace(go.Scatter(x=NUM_CARACT, y=compararPCA,
                    name='PCA',
                    line=dict(color='firebrick', width=4,
                              dash='dash')))
figCompararNumCaract.write_image("./graficos/figCompararNumCaract.png")
