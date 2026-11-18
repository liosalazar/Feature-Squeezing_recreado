# SwinSR-Defense — reproducción piloto

Este repositorio reúne el código del piloto local presentado en Semana 13. Ejecuta una parte del flujo del artículo: generar un ejemplo MI-FGSM, ajustar SwinIR con una pareja adversaria–limpia y volver a clasificar la reconstrucción con ResNet-50.

## Resultado documentado

- Imagen: autobús escolar de ImageNet, 224 × 224.
- Clasificador: ResNet-50 con pesos TorchVision IMAGENET1K_V2.
- Ataque: MI-FGSM, ε = 5/255, 10 pasos.
- Resultado: `school bus` → `sundial` → `school bus` en la muestra del piloto.
- Reconstruidor: arquitectura SwinIR del cuaderno público de los autores; ajuste durante 35 épocas con una pareja y pérdida MSE.
- Pérdida de entrenamiento: 0.1030277 → 0.0028985.

**Alcance:** demostración de una muestra. La pareja mostrada se usó también durante el ajuste. El resultado no mide generalización, exactitud robusta ni constituye una reproducción completa del artículo.

## Ejecutar

1. Instala Python y una combinación compatible de PyTorch y torchvision para tu equipo. Se usó Python 3.13, PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126 y una RTX 3050 de 6 GB. CPU también es posible, pero será más lento.
2. Instala `numpy` y `Pillow` además de PyTorch y torchvision.
3. Coloca tus copias locales autorizadas de las imágenes en `data/` con estos nombres:
   - `school_bus_source.png`
   - `toaster_source.png`
4. Descarga el cuaderno de arquitectura de los autores desde el repositorio enlazado abajo y guárdalo como `third_party/SwinSR-Defense_Autores.ipynb`. No está incluido aquí.
5. Desde la raíz del repositorio, ejecuta:

```powershell
python .\src\01_mifgsm_attack.py
python .\src\02_train_swinir.py
```

El primer script descarga automáticamente los pesos de ResNet-50 la primera vez, genera los ejemplos y guarda `results/attack_results.json`. El segundo carga la arquitectura SwinIR del cuaderno de autores, entrena el piloto y guarda el checkpoint local, la curva CSV y los resultados. El checkpoint y las imágenes no se suben a Git.

## Estructura

- `src/01_mifgsm_attack.py`: crea los ejemplos y registra las predicciones.
- `src/02_train_swinir.py`: ajusta SwinIR y clasifica la reconstrucción.
- `results/`: métricas registradas de la corrida ya ejecutada.
- `data/`: datos locales; no se versionan.
- `third_party/`: espacio para el cuaderno consultado de los autores; no se versiona.

## Procedencia

Este trabajo se basa en Gulsoy, E. K., Gulsoy, T., Ayas, S., & Baykal Kablan, E. (2026), *SwinSR-Defense: Swin transformer with super-resolution capability for adversarial robustness*, *Applied Soft Computing, 203*, 116135. https://doi.org/10.1016/j.asoc.2026.116135

Repositorio de código original: https://github.com/TolgahanGulsoy/SwinSR-Defense

Los archivos de arquitectura y datos de terceros no se redistribuyen en este repositorio. Consulta sus fuentes y condiciones antes de reutilizarlos.
