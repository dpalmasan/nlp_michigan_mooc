#!/bin/bash

# Ejecuta el scorer con el baseline
./scorer2 English.baseline data/English-dev.key data/English.sensemap

file="KNN-English.answer"

if [ -f "$file" ]
then
	echo "Se obtuvieron los modelos, proceder al siguiente paso..."
else
	# Corre los experimentos para los distintos lenguajes y crea archivos de salida
    echo "Corriendo Experimentos y generando archivos de salida"
    python main.py data/English-train.xml data/English-dev.xml KNN-English.answer SVM-English.answer Best-English.answer english
    python main.py data/Spanish-train.xml data/Spanish-dev.xml KNN-Spanish.answer SVM-Spanish.answer Best-Spanish.answer spanish
    python main.py data/Catalan-train.xml data/Catalan-dev.xml KNN-Catalan.answer SVM-Catalan.answer Best-Catalan.answer catalan
fi



# Ejecutar scorer para ver resultados
echo "-----------------------"
echo "Para English"
echo "-----------------------"

./scorer2 KNN-English.answer data/English-dev.key data/English.sensemap
./scorer2 SVM-English.answer data/English-dev.key data/English.sensemap
./scorer2 Best-English.answer data/English-dev.key data/English.sensemap

echo "-----------------------"
echo "Para Spanish"
echo "-----------------------"

./scorer2 KNN-Spanish.answer data/Spanish-dev.key
./scorer2 SVM-Spanish.answer data/Spanish-dev.key
./scorer2 Best-Spanish.answer data/Spanish-dev.key

echo "-----------------------"
echo "Para Catalan"
echo "-----------------------"

./scorer2 KNN-Catalan.answer data/Catalan-dev.key
./scorer2 SVM-Catalan.answer data/Catalan-dev.key
./scorer2 Best-Catalan.answer data/Catalan-dev.key
