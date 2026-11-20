# Feature Squeezing en ImageNet

MobileNet-v1 se evaluó con el checkpoint publicado y las primeras 200 imágenes de validación, ordenadas por su identificador original. El modelo acertó 139 de 200 (69,50 %). La huella `a2749` de las primeras 100 imágenes clasificadas correctamente coincide con la selección de los autores.

Con las nueve variantes de entrada publicadas, la exactitud media fue de 2,78 % sin tratamiento y de 68,11 % después del filtro mediano de 2 × 2. Ambos valores coinciden con la Tabla 3. En las 200 imágenes limpias, la exactitud bajó de 69,50 % a 65,00 % tras aplicar el mismo filtro.

| Detector conjunto en ImageNet | Ejecución local | Tabla 5 |
|---|---:|---:|
| Umbral | 1,1705 | 1,2128 |
| Detección de ejemplos adversarios exitosos | 87,38 % (374/428) | 85,94 % |
| Falsas alertas en imágenes limpias | 10,19 % (11/108) | 8,33 % |
| Área bajo la curva ROC para ejemplos exitosos y limpios | 94,50 % | 94,24 % |

Se calibró el umbral con 92 imágenes limpias de la partición original del repositorio y se midieron falsas alertas en otras 108. Al aplicar el umbral publicado a los mismos puntajes limpios de evaluación, las falsas alertas bajaron a 9/108 (8,33 %). Esta comprobación muestra el efecto del umbral sobre dos alertas. La causa de la diferencia entre los umbrales sigue abierta.

La réplica usó PyTorch y reevaluó los ejemplos del repositorio original. La coincidencia de la selección y de varias cifras demuestra la fidelidad de esas partes del protocolo. Quedan pendientes verificaciones para afirmar equivalencia completa con el artículo.

[Artículo de Xu, Evans y Qi (2018), tablas 3 y 5](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-4_Xu_paper.pdf). [Protocolo del repositorio de los autores](https://github.com/mzweilin/EvadeML-Zoo/blob/4dbeee04874a836f79782802b2bf8a8612a2a86f/Reproduce_FeatureSqueezing.md).
