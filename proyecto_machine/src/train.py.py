# GUARDARE EN EL MISMO ARCHIVO LOS TRES MODELOS OBTENIDOS
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBRFClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# Primero leemos el dataset
calidad = pd.read_csv('waterQuality1.csv')
calidad['is_safe'] = pd.to_numeric(calidad['is_safe'], errors='coerce') # LOS CAMBIAMOS A NAN PARA ELIMINARLOS
calidad1 = calidad.dropna(subset=['is_safe'])
# Y TENGO QUE CAMBIAR EL AMONIACO A NUMERICO, QUE ES LA UNICA OTRA COLUMNA QUE ESTA COMO OBJETO
calidad1['ammonia'] = pd.to_numeric(calidad1['ammonia'], errors='coerce')

# Extraemos X e Y
X = calidad1.drop('is_safe',axis=1)
y = calidad1['is_safe']

# normalizamos, ANTES QUE NADA
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
X = scaler.fit_transform(X)

smote = SMOTE() # RESAMPLEAMOS
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print("Total features shape:", X.shape) # ANTES DEL RESAMPLEO
print("Train features shape:", X_train.shape)
print("Train target shape:", y_train.shape)
print("Test features shape:", X_test.shape)
print("Test target shape:", y_test.shape)

# Primero, random forest
model_rf_resampleado = RandomForestClassifier(n_estimators=100,max_features=3, random_state=42)
model_rf_resampleado.fit(X_train, y_train)
pred_rf_resampleado = model_rf_resampleado.predict(X_test)
model_rf_val_res = RandomForestClassifier(n_estimators=100, max_features=3, random_state=42)
results_cv_rf_res = cross_val_score(model_rf_val_res, X_train, y_train, cv=10, scoring='recall')
print(results_cv_rf_res)
print(results_cv_rf_res.mean())
print(classification_report(y_test,pred_rf_resampleado)) # PARA VER todas las metricas
sns.heatmap(confusion_matrix(y_test, pred_rf_resampleado, normalize='true'), annot=True) # Y LA MATRIZ DE CONFUSION

# y guardamos el modelo
with open('my_model_RANDOM_FOREST.pkl', 'wb') as file:
    pickle.dump(model_rf_resampleado, file)


# AHORA, CON XGBOOST
model_XGB_resampleado = XGBRFClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
results_cv_xgb_resampleado = cross_val_score(model_XGB_resampleado, X_train, y_train, cv=10, scoring='recall')
print(results_cv_xgb_resampleado)
print(results_cv_xgb_resampleado.mean()) 
model_XGB_resampleado.fit(X_train, y_train)
pred_xgb_resampleado = model_XGB_resampleado.predict(X_test)
print(classification_report(y_test,pred_xgb_resampleado)) # METRICAS
sns.heatmap(confusion_matrix(y_test, pred_xgb_resampleado, normalize='true'), annot=True)

# Y GUARDAMOS EL MODELO
with open('my_model_XGBOOST.pkl', 'wb') as file:
    pickle.dump(model_XGB_resampleado, file)
    
# Y FINALMENTE EL NO SUPERVISADO
calidad_sin_target = calidad1.drop(columns=['is_safe']).dropna() 
pca = PCA(n_components=2) 
componentes_principales = pca.fit_transform(calidad_sin_target)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_) 
plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=componentes_principales.columns)
plt.title('Heatmap de Loadings de PCA')
plt.xlabel('Características Originales')
plt.ylabel('Componentes Principales')
plt.show()

columnas_a_borrar = [2, 4, 7, 8, 9, 10, 11, 12, 14, 17, 18, 19]
calidad_filtrada = calidad_sin_target.drop(columns=calidad_sin_target.columns[columnas_a_borrar])
calidad_normalizada = scaler.fit_transform(calidad_filtrada)
calidad_filtrada = pd.DataFrame(calidad_normalizada, columns=calidad_filtrada.columns)
kmeans_calidad = KMeans(n_clusters=2)  # incluimos los 2 clusters
clusters_calidad = kmeans_calidad.fit_predict(calidad_filtrada)
calidad_clusterizada = calidad_filtrada.copy()  # hago una copia
calidad_clusterizada['cluster'] = clusters_calidad# asi, tenemos las etiquetas de los cluster en cada columna

grouped_data1 = calidad_clusterizada.groupby('cluster') # es necesario agrupar, SI NO, NO VERIAMOS LOS GRUPOS

for cluster, group in grouped_data1: # PARA VER LAS CARACTERISTICAS
    print(f"Cluster {cluster}:")
    print(group.describe())
    print("\n")
calidad_clusterizada.boxplot(by='cluster', figsize=(10, 6)) #  SI QUIERO VERLO EN GRAFICOS, un boxplot de cada característica por cluster:
plt.show()

# y guardamos el modelo en un archivo
with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans_calidad, file)
