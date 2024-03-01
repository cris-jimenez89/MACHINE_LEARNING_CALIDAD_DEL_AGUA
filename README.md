# MACHINE_LEARNING_CALIDAD_DEL_AGUA
* Desarrollamos un proyecto de machine learning, para determinar la calidad del agua y su potabilidad, con la columna target IS_SAFE, sobre si el agua se considera segura o no. El proyecto en principio se hará
  con aprendizaje supervisado de clasificacion, al ser nuestro target un clasificador binario (segura o no segura). Usaremos tanto modelos normales (regresion logistica, SVM..) como modelos con técnicas de más complejas de ensembles 
  (XGBOOST, RANDOM FOREST...). 
* Descargamos los datos de kaggle, y nos servimos de dos datasets, uno sobre la calidad del agua, con columnas de las concentraciones de distintas sustancias que pueden ser contaminantes y otro sobre su potabilidad, con features como el 
  pH, la dureza, la turbicidad...
* Dado el desbalance del target, haremos modelos con oversampling y sin balancear, y los compararemos.
* Buscaremos los parametros óptimos de nuestros mejores modelos, con gridsearch y cross value.
* Damos una vuelta al modelo supervisado, que finalmente solo determinaria si esta contaminada o no segun sus componentes, Y YA LAS FEATURES TE APORTAN LA MAYOR PARTE DE LA INFORMACION Y ES MENOS PREDICTIVO, Y USAMOS APRENDIZAJE NO SUPERVISADO, PARA IDENTIFICAR PATRONES OCULTOS E INFORMACION NUEVA, ASI COMO POSIBLES RELACIONES ENTRE FEATURES QUE NO SE HABIAN CONSIDERADO CON ANTERIORIDAD
* Keywords(Python, kaggle, Data Cleaning, Visualización, aprendizaje supervisado, aprendizaje no supervisado,Kmeans)
![image](https://github.com/cris-jimenez89/MI-PORTFOLIO-DE-DATA/assets/145456716/e0544f16-8e65-4891-8302-b4eb58caac9d)
![image](https://github.com/cris-jimenez89/MI-PORTFOLIO-DE-DATA/assets/145456716/3b6667a6-0323-4522-922c-354787306169)
![newplot](https://github.com/cris-jimenez89/MI-PORTFOLIO-DE-DATA/assets/145456716/44c813a3-3c7f-4611-9f05-47ea97583db6)
