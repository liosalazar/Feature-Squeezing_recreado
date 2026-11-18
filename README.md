# Feature Squeezing — reproducción acotada

Proyecto de reproducción de un experimento del artículo de Xu, Evans y Qi presentado en NDSS 2018. Reproduce el protocolo de MNIST con la CNN de siete capas, FGSM con ε = 0.3 y reducción de profundidad a 1 bit. La réplica se implementa en PyTorch a partir de la arquitectura y configuración publicadas en el artículo y en el código original.

## Qué implementa

1. Entrena en MNIST el modelo con la arquitectura y los hiperparámetros del repositorio original de los autores.
2. Genera ejemplos FGSM sobre las imágenes de prueba con ε = 0.3; recorta los píxeles a [0, 1] y los cuantiza a 8 bits, como especifica la sección experimental del artículo.
3. Compara la clasificación de la imagen atacada antes y después del filtro binario de 1 bit.
4. Calcula la puntuación L1 entre las distribuciones softmax de la imagen y su versión filtrada.
5. Usa 500 pares limpios/adversarios para calibración y 500 pares para validación. El umbral se fija con imágenes limpias de calibración para buscar una tasa de falsos positivos de 5%.
6. Guarda métricas y una visualización limpia → FGSM → filtrada.

Esta es una reproducción acotada de las pruebas FGSM/1-bit del artículo, no de sus once variantes de ataque ni de CIFAR-10 e ImageNet. Se eligen los primeros 100 ejemplos de prueba clasificados correctamente, como indica el protocolo para la tabla de robustez. La corrida local alcanzó 99.25% de exactitud en todo MNIST; en los 100 ejemplos seleccionados, 3% se clasificaron correctamente bajo FGSM y 92% después del filtro de 1 bit. El artículo reporta 99.43% de exactitud limpia, 54% bajo FGSM y 92% con el filtro; en imágenes limpias filtradas reporta 99.33% (Tablas 1 y 3). La exactitud tras el filtro coincide, pero la exactitud bajo FGSM difiere. Por ello, no afirmamos una reproducción numérica completa: se reproduce el flujo y los parámetros de esa prueba con pesos reentrenados localmente. La corrida incluye esa diferencia para su análisis.

## Ejecución

Desde la raíz del proyecto, instala las dependencias y ejecuta:

```powershell
python -m pip install -r requirements.txt
python .\src\replicate_mnist_fgsm.py
```

La primera corrida descarga MNIST y entrena el modelo por 100 épocas. Las corridas siguientes reutilizan el checkpoint guardado localmente. Se usa GPU CUDA cuando está disponible; también se puede forzar CPU con `--device cpu`.

## Resultados generados

- `results/metrics.json`: resultados, parámetros y alcance de la reproducción.
- `results/metrics.csv`: métricas en formato tabular.
- `results/training_history.csv`: pérdida y exactitud por época.
- `results/mnist_cnn_paper_architecture.pt`: checkpoint local; no subir a Git.
- `results/mnist_fgsm_feature_squeezing.png`: ejemplo visual.

En la corrida local, la exactitud limpia de prueba fue 99.25%. En los 100 ejemplos de robustez seleccionados, los aciertos bajo FGSM fueron 3% sin el filtro y 92% tras el filtro. Para el detector, se calibró con 500 imágenes limpias; en validación se obtuvo FPR de 6.8%, tasa de detección de ataques exitosos de 100% y AUC de 0.999972. La figura con el 7 muestra un caso individual, no el resultado agregado.

## Fuentes primarias

- Xu, W., Evans, D., & Qi, Y. (2018). *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks*. Network and Distributed System Security Symposium (NDSS).
- Artículo completo: https://arxiv.org/abs/1704.01155
- Repositorio original: https://github.com/QData/FeatureSqueezing
- Arquitectura MNIST de CleverHans v1.0.0 utilizada por el artículo: https://github.com/cleverhans-lab/cleverhans/blob/v1.0.0/cleverhans/utils_mnist.py
- Figura 1 original: incluida en la presentación con atribución al artículo.

## Nota de fidelidad

La implementación traslada a PyTorch la arquitectura, normalización, suavizado de etiquetas, optimizador, hiperparámetros, filtro de 1 bit y construcción FGSM documentados en el artículo y el repositorio original. El artículo indica que las imágenes adversarias se recortan a [0,1] y cuantizan a 8 bits; el helper publicado no muestra ese paso, así que la réplica sigue la descripción del artículo. Se moderniza TensorFlow/Keras de 2018 a PyTorch para ejecutar el protocolo en el entorno disponible. Se entrena localmente la CNN con la configuración del paper; no se usa el checkpoint de los autores. El artículo publica código y enlace de descarga del modelo, pero el enlace del checkpoint respondió con acceso denegado en esta sesión. El SwinIR y sus resultados no se reutilizan.
