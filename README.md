# Feature Squeezing

La evaluación con los checkpoints de referencia abarcó MNIST, CIFAR-10 e ImageNet. La [comparación de ImageNet con el artículo](results/official_imagenet/COMPARACION.md) reúne las métricas para el pitch, el tamaño de cada muestra y la comprobación del umbral.

## Experimento exploratorio anterior

La documentación desde aquí describe la primera ejecución con una CNN entrenada localmente. La evaluación con los checkpoints publicados es una ejecución posterior.

Este proyecto reproduce parcialmente el método de Xu, Evans y Qi (NDSS 2018). El experimento implementado cubre MNIST, el ataque FGSM y la reducción de profundidad de color a 1 bit. También incluye una evaluación del detector basado en la diferencia L1 entre predicciones. No es una reproducción completa del artículo.

## Qué hace el experimento

1. Entrena en MNIST una CNN de siete capas basada en la arquitectura publicada, durante 100 épocas, con lotes de 128, Adadelta y suavizado de etiquetas.
2. Selecciona las primeras 100 imágenes de prueba que el modelo clasifica correctamente.
3. Aplica FGSM con ε = 0,3, limita los píxeles al intervalo [0, 1] y cuantiza las imágenes a 8 bits.
4. Compara la clasificación de las imágenes atacadas antes y después de reducir su profundidad de color a 1 bit.
5. Ejecuta además el detector FGSM con ε = 0,1, 0,2 y 0,3. Usa la primera mitad de los datos para seleccionar el umbral y la segunda mitad para validarlo.

## Resultados de esta ejecución

- Exactitud limpia en todo el conjunto de prueba: 99,25 % (el artículo reporta 99,43 %).
- En las 100 imágenes seleccionadas: exactitud con FGSM ε = 0,3 de 3 % antes del filtro y 92 % después de la reducción a 1 bit.
- El artículo reporta 54 % antes del filtro y 92 % después. La exactitud local sin filtro no coincide con el artículo; no se modificó el ataque para forzar la coincidencia.
- En ε = 0,5, la exactitud cae a 1 % sin filtro y 4 % después del filtro. La cuantización puede ayudar con ciertos niveles de perturbación, pero no garantiza recuperar la clase ante perturbaciones fuertes.
- Detector: umbral 0,34356; falsos positivos en imágenes limpias de validación 0,44 %; detección de ataques exitosos ε = 0,3 de 99,87 % (4 689 de 4 695 ataques exitosos entre 5 000 entradas); AUC-ROC local 0,99988. Este AUC no es directamente comparable con el AUC conjunto del artículo.

Los resultados dependen del modelo entrenado localmente. No se obtuvo el checkpoint original de los autores; por ello se reentrenó la arquitectura en PyTorch y la diferencia de robustez frente al artículo sigue sin explicación definitiva.

## Alcance pendiente frente al artículo

El artículo evalúa MNIST, CIFAR-10 e ImageNet, varias transformaciones y once variantes de ataque, incluidos FGSM, BIM, DeepFool y ataques dirigidos. Esta reproducción solo cubre MNIST, FGSM y el filtro de 1 bit, además del detector FGSM. Quedan fuera CIFAR-10, ImageNet, los otros ataques, el suavizado mediano y *non-local means*. Se usa PyTorch en lugar del stack original de TensorFlow/Keras.

## Ejecución en Windows

Instala Python 3.10 o posterior y ejecuta `EJECUTAR_EXPERIMENTO.bat`. El archivo crea un entorno virtual local e instala las dependencias si hace falta. La primera ejecución descarga MNIST y entrena durante 100 épocas; puede tardar. Después reutiliza `results/mnist_cnn_paper_architecture.pt` si el checkpoint está disponible.

También se puede ejecutar desde PowerShell:

```powershell
python -m pip install -r requirements.txt
python .\src\replicate_mnist_fgsm.py
```

Para limitar el número de ejemplos del experimento del detector, usa `--sample-count N`.

## Archivos principales

- `src/replicate_mnist_fgsm.py`: arquitectura, entrenamiento, ataque, filtro y mediciones.
- `results/metrics.json` y `results/metrics.csv`: resultados numéricos.
- `results/mnist_fgsm_feature_squeezing.png`: ejemplo visual de una imagen limpia, atacada y filtrada.
- `results/mnist_fgsm_epsilon_sweep.png`: exactitud antes y después del filtro para ε entre 0 y 1.
- `results/training_history.csv`: métricas de entrenamiento por época.
- `results/mnist_cnn_paper_architecture.pt`: checkpoint local; no representa pesos oficiales de los autores.
- `assets/workflow_fig1_crop.png`: recorte de la figura de workflow del artículo.

## Fuentes

- Xu, W., Evans, D., & Qi, Y. (2018). *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks*. NDSS 2018. https://dev.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-4_Xu_paper.pdf
- Repositorio original de los autores: https://github.com/QData/FeatureSqueezing
- Arquitectura MNIST de CleverHans v1.0.0: https://github.com/cleverhans-lab/cleverhans/blob/v1.0.0/cleverhans/utils_mnist.py


Revisión del 19/11/2026. Se corrigió el cálculo de AUC para empates y el texto dinámico de decisión de la figura. El AUC guardado corresponde a la corrida anterior y está pendiente de recálculo. Se verificaron sintaxis y cuatro casos de AUC. La ejecución completa de PyTorch sigue pendiente en esta PC.
