# Subir este repo a GitHub

El repositorio Git local ya está inicializado en la carpeta `SwinSRDefense_GitHub` y tiene un commit inicial. Para publicar el enlace solicitado por la plantilla:

1. Entra a GitHub y crea un repositorio vacío llamado, por ejemplo, `SwinSRDefense-Piloto`. No marques las opciones para generar README, licencia o `.gitignore`, porque esos archivos ya están preparados localmente.
2. Abre PowerShell en la carpeta `SwinSRDefense_GitHub` y ejecuta los comandos siguientes, reemplazando `<tu-usuario>` por tu nombre de GitHub:

```powershell
git remote add origin https://github.com/<tu-usuario>/SwinSRDefense-Piloto.git
git push -u origin main
```

3. Abre el enlace del repositorio en el navegador y confirma que aparecen `README.md`, `src/` y `results/`.
4. Copia ese enlace propio a la diapositiva de experimentación. El enlace actual al repo de los autores es una referencia a su implementación, no sustituye el tuyo.

El `.gitignore` excluye imágenes de entrada, PNG de salida, pesos `.pth`, caché de modelos y el cuaderno de terceros. No subas tokens, claves, contraseñas ni datos que no tengas permiso para compartir. Para que otra persona ejecute el piloto, debe poner imágenes autorizadas en `data/` y bajar por separado el cuaderno de los autores.
