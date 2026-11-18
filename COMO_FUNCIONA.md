# Qué hace cada archivo

## `src/01_mifgsm_attack.py`

1. Carga ResNet-50 con pesos preentrenados de ImageNet.
2. Lee las imágenes locales de `data/`, las convierte a RGB y las prepara en 224 × 224.
3. Clasifica la imagen limpia y guarda su etiqueta Top-1.
4. Ejecuta MI-FGSM durante 10 pasos. Cada paso usa el gradiente de la pérdida del clasificador para cambiar píxeles; la perturbación queda limitada a 5/255 y los valores siguen en el rango válido de imagen.
5. Clasifica la imagen adversaria y calcula la perturbación máxima observada.
6. Guarda las predicciones y parámetros en `results/attack_results.json`, además de las imágenes locales generadas.

La tostadora sirve como segunda comprobación de entrada, pero ResNet-50 no acertó su clase limpia en esta corrida. Por ese motivo no cuenta como un ataque exitoso sobre una entrada inicialmente bien clasificada.

## `src/02_train_swinir.py`

1. Abre el cuaderno público de los autores colocado en `third_party/` y carga la definición de SwinIR de su segunda celda.
2. Lee del paso anterior la imagen limpia y su versión MI-FGSM.
3. Entrena el reconstruidor durante 35 épocas con esa única pareja: adversaria como entrada y limpia como objetivo. Usa AdamW y MSE.
4. Reconstruye la entrada adversaria y la clasifica con ResNet-50.
5. Guarda el checkpoint local, la pérdida por época, las etiquetas y la configuración usada.

**Límite del piloto:** se usa la misma pareja para entrenar y mostrar el resultado. La caída de MSE y la etiqueta recuperada describen esa muestra; no demuestran que la reconstrucción funcione en imágenes nuevas.

## Archivos de resultados

- `results/attack_results.json`: etiquetas limpia/adversaria, presupuesto y norma observada del ataque.
- `results/training_results.json`: configuración, pérdida inicial/final y etiqueta antes/después de SwinIR.
- `results/training_loss.csv`: pérdida de entrenamiento por época.

Los PNG y el checkpoint que se generan al correr los scripts quedan locales y están excluidos de Git.
