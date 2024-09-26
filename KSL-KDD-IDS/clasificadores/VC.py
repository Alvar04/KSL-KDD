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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle

#INICIO DEL PROGRAMA
print("EVALUACIÓN DEL CLASIFICADOR - " + sys.argv[0] )




#DECLARACIÓN CONSTANTES
L_CLF = 'VC'




#LECTURA DE ARGUMENTOS
num_caract = sys.argv[1]
selector = sys.argv[2]
print("ARGUMENTOS: num_caract=" + str(num_caract) + " selector=" + str(selector))




#APERTURA FICHERO
print ("LECTURA DE FICHERO")
X_train = np.loadtxt("./datos/features" + str(num_caract) + "selector" + str(selector) + "/X_train.csv", delimiter=',')
y_train = np.loadtxt("./datos/features" + str(num_caract) + "selector" + str(selector) + "/y_train.csv", delimiter=',')
X_test = np.loadtxt("./datos/features" + str(num_caract) + "selector" + str(selector) + "/X_test.csv", delimiter=',')
y_test = np.loadtxt("./datos/features" + str(num_caract) + "selector" + str(selector) + "/y_test.csv", delimiter=',')
le = pickle.load(open('./modelos/le.sav', 'rb'))




#SELECCION DE CLASIFICADOR
clf1 = tree.DecisionTreeClassifier()
clf2 = naive_bayes.ComplementNB()
clf3 = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(int(num_caract)), random_state=1, max_iter=10000)
eclf = VotingClassifier(estimators=[('dt', clf1),  ('nb', clf2), ('ann', clf3)],voting='soft',n_jobs=-1)


#PREPROCESAMIENTO
print("PREPROCESAMIENTO")
#Normalizamos
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

#Escalamos
scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




#FASE DE ENTRENAMIENTO
print ("FASE DE ENTRENAMIENTO")
t_inicio_entrenamiento = time.time()
eclf.fit(X_train, y_train)
t_fin_entrenamiento = time.time()

print ("Training time: " + str(t_fin_entrenamiento - t_inicio_entrenamiento))



#FASE DE CLASIFICACIÓN
print ("FASE DE CLASIFICACIÓN")
t_inicio_clasif = time.time()
y_pred = eclf.predict(X_test)
t_fin_clasif = time.time()

print ("Testing time: " + str(t_fin_clasif - t_inicio_clasif))





#OBTENCIÓN DE MÉTRICAS DE RENDIMIENTO
print ("OBTENCIÓN DE MÉTRICAS DE RENDIMIENTO")
p, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
bacc = balanced_accuracy_score(y_pred, y_test)

y_test = le.inverse_transform(y_test.astype(int))
y_pred = le.inverse_transform(y_pred.astype(int))
v_p, v_recall, v_fscore, v_support = precision_recall_fscore_support(y_test, y_pred, average=None, labels = le.classes_)

cnf_matrix = confusion_matrix(y_test, y_pred)

#Imprimir resumen por pantalla
print ("Exactitud balanceada (Balanced accuracy score): ")
print (bacc)
print (classification_report(y_test,y_pred))




#GUARDAR RESULTADOS A DISCO
print ("GUARDAR RESULTADOS A DISCO")
if not os.path.exists("./resultados"):
    os.mkdir("./resultados")

#Métricas de rendimiento globales
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-bacc.dat',"w") as f:
    f.write(str(float(bacc)))
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-precision.dat',"w") as f:
    f.write(str(float(p)))
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Tiempo_entrenamiento.dat',"w") as f:
    f.write(str(float(t_fin_entrenamiento - t_inicio_entrenamiento)))
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Tiempo_clasificacion.dat',"w") as f:
    f.write(str(float(t_fin_clasif - t_inicio_clasif)))

#Metricas de rendimiento por clase
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Precision.npy','wb') as f:
    np.save(f, v_p)
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Sensibilidad.npy','wb') as f:
    np.save(f, v_recall)
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Fscore.npy','wb') as f:
    np.save(f, v_fscore)
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Total.npy','wb') as f:
    np.save(f, v_support)


#Métricas para la selección de características
with open('./resultados/descr_general.dat',"a") as f:
    f.write(L_CLF + " num_caract=" + str(num_caract) + " selector=" + str(selector) + " bacc=" + str(float(bacc)) + " p=" + str(float(p)) + " TRT=" + str(t_fin_entrenamiento - t_inicio_entrenamiento) + " \n")
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/bacc_FeaturesSelector.dat',"a") as f:
    f.write(str(float(bacc)) + " \n")


#Confusion Matrix y métricas derivadas
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-confusion_matrix.npy','wb') as f:
    np.save(f, confusion_matrix)
