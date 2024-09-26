import os.path
import sys
import numpy as np
import csv
import time
from sklearn import tree
from sklearn import svm
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle





#DECLARACIÓN CONSTANTES
NUM_CARACT = [5,10,15,20,25]
CLASIFICADORES = ['DT','NB','ANN','SVM','RF','GBM','VC']

#APERTURA FICHERO
print ("LECTURA DE FICHERO")
reader = csv.reader(open("./datos/NSL-KDD.csv"), delimiter=",")
raw_data = list(reader)

np_data = np.asarray(raw_data, dtype=None)

X = np_data[:, 0:-2]  # Seleccionar todas las columnas menos las dos últimas
y = np_data[:, -2]   # Seleccionar la penúltima columna (etiqueta como cadena)





#AGRUPAMOS ETIQUETAS
print("AGRUPAMOS ETIQUETAS")
for i in range(0,len(y)):
    if (y[i] in ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']):
        y[i] = 'Probe'
    elif (y[i] in ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named']):
        y[i] = 'R2L'
    elif (y[i] in ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']):
        y[i] = 'U2R'
    elif (y[i] in ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'worm', 'mailbomb']):
        y[i] = 'DoS'
    elif (y[i] == 'normal'):
        y[i] = 'Normal'
    else:
        y[i] = 'Unknown'





#PREPROCESAMIENTO
print("PREPROCESAMIENTO")

#Preprocesamos el array para que únicamente contega valores normalizados de tipo float32
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1].astype(str))
X[:,2] = le.fit_transform(X[:,2].astype(str))
X[:,3] = le.fit_transform(X[:,3].astype(str))
y = le.fit_transform(y)

X = X.astype(np.float64)
y = y.astype(np.float64)

#Eliminamos NAN
X = np.nan_to_num(X.astype(np.float64))





#DIVISIÓN DEL DATASET
print ("DIVISIÓN DEL DATASET")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print ("X_train, y_train:", X_train.shape, y_train.shape)
print ("X_test, y_test:", X_test.shape, y_test.shape)

# RESUMEN DE LOS DATOS Y GUARDADO EN DISCO
unique_elements, counts_elements = np.unique(le.inverse_transform(y_train.astype(np.int64)), return_counts=True)
print ("Número de elementos de cada clase en el Train Set:")
print(np.asarray((unique_elements, counts_elements)))
with open("./datos/train_descr.txt","w") as f:
    f.write(str(np.asarray((unique_elements, counts_elements))))

unique_elements, counts_elements = np.unique(le.inverse_transform(y_test.astype(np.int)), return_counts=True)
print ("Número de elementos de cada clase en el Test Set:")
print(np.asarray((unique_elements, counts_elements)))
with open("./datos/test_descr.txt","w") as f:
    f.write(str(np.asarray((unique_elements, counts_elements))))





for num in NUM_CARACT:
    # SELECCION DE CARACTERÍSTICAS
    print ("SELECCION DE CARACTERÍSTICAS: " + str(num) + " CARACTERÍSTICAS")
    estimador = tree.DecisionTreeClassifier()
    selector1 = RFE(estimador, n_features_to_select=int(num), step=1)
    selector2 = PCA(n_components=int(num))

    print ("SELECCION DE CARACTERÍSTICAS: " + str(num) + " CARACTERÍSTICAS, " + "SELECTOR RFE")
    selector1 = selector1.fit(X_train, y_train)
    print (selector1.ranking_)

    print ("SELECCION DE CARACTERÍSTICAS: " + str(num) + " CARACTERÍSTICAS, " + "SELECTOR PCA")
    selector2 = selector2.fit(X_train, y_train)


    X_train1 = selector1.transform(X_train)
    X_test1 = selector1.transform(X_test)
    X_train2 = selector2.transform(X_train)
    X_test2 = selector2.transform(X_test)

    # CREAR DIRECTORIOS PARA GUARDAR DATOS PROCESADOS
    if not os.path.exists("./datos/features" + str(num) + "selectorRFE"):
        os.mkdir("./datos/features" + str(num) + "selectorRFE")
    if not os.path.exists("./datos/features" + str(num) + "selectorPCA"):
        os.mkdir("./datos/features" + str(num) + "selectorPCA")

    #CREAR DIRECTORIOS PARA GUARDAR RESULTADOS
    if not os.path.exists("./resultados"):
        os.mkdir("./resultados")
    if not os.path.exists("./resultados/features" + str(num) + "selectorRFE"):
        os.mkdir("./resultados/features" + str(num) + "selectorRFE")
    if not os.path.exists("./resultados/features" + str(num) + "selectorPCA"):
        os.mkdir("./resultados/features" + str(num) + "selectorPCA")

    #CREAR DIRECTORIOS PARA GUARDAR GRÁFICAS
    if not os.path.exists("./graficos"):
        os.mkdir("./graficos")
    if not os.path.exists("./graficos/features" + str(num) + "selectorRFE"):
        os.mkdir("./graficos/features" + str(num) + "selectorRFE")
    if not os.path.exists("./graficos/features" + str(num) + "selectorPCA"):
        os.mkdir("./graficos/features" + str(num) + "selectorPCA")


    #GUARDAR EN DISCO
    np.savetxt("./datos/features" + str(num) + "selectorRFE/X_train.csv", X_train1, delimiter=',')
    np.savetxt("./datos/features" + str(num) + "selectorRFE/y_train.csv", y_train, delimiter=',')
    np.savetxt("./datos/features" + str(num) + "selectorRFE/X_test.csv", X_test1, delimiter=',')
    np.savetxt("./datos/features" + str(num) + "selectorRFE/y_test.csv", y_test, delimiter=',')
    with open("./datos/features" + str(num) + "selectorRFE/ranking.npy",'wb') as f:
        np.save(f, selector1.ranking_)


    np.savetxt("./datos/features" + str(num) + "selectorPCA/X_train.csv", X_train2, delimiter=',')
    np.savetxt("./datos/features" + str(num) + "selectorPCA/y_train.csv", y_train, delimiter=',')
    np.savetxt("./datos/features" + str(num) + "selectorPCA/X_test.csv", X_test2, delimiter=',')
    np.savetxt("./datos/features" + str(num) + "selectorPCA/y_test.csv", y_test, delimiter=',')


#CREAR DIRECTORIO PARA GUARDAR MODELOS
if not os.path.exists("./modelos"):
    os.mkdir("./modelos")
pickle.dump(le, open('./modelos/le.sav', 'wb'))





#TESTEAR LOS CLASIFICADORES
print("TESTEAR LOS CLASIFICADORES")
if os.path.exists('./resultados/descr_general.dat'):
  os.remove('./resultados/descr_general.dat')
for num in NUM_CARACT:
    if os.path.exists("./resultados/features" + str(num) + "selectorPCA/bacc_FeaturesSelector.dat"):
      os.remove("./resultados/features" + str(num) + "selectorPCA/bacc_FeaturesSelector.dat")
    if os.path.exists("./resultados/features" + str(num) + "selectorRFE/bacc_FeaturesSelector.dat"):
      os.remove("./resultados/features" + str(num) + "selectorRFE/bacc_FeaturesSelector.dat")

for clf in CLASIFICADORES:
    for num in NUM_CARACT:
        script_descriptor = open("./clasificadores/" + clf + ".py")
        script = script_descriptor.read()
        sys.argv = [str(clf) + ".py", int(num), 'RFE']
        exec(script)
        sys.argv = [str(clf) + ".py", int(num), 'PCA']
        exec(script)


#EXTRAER GRÁFICAS
print("EXTRAER GRÁFICAS")
for num in NUM_CARACT:
    script_descriptor = open("./ProcesarResultados.py")
    script = script_descriptor.read()
    sys.argv = ["ProcesarResultados.py", str(num), 'RFE']
    exec(script)
for num in NUM_CARACT:
    script_descriptor = open("./ProcesarResultados.py")
    script = script_descriptor.read()
    sys.argv = ["ProcesarResultados.py", str(num), 'PCA']
    exec(script)
