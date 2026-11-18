# Actualizar el repositorio publicado

El proyecto ya está publicado en https://github.com/liosalazar/SwinSRDefense-Piloto y la rama local `main` sigue a `origin/main`.

Después de editar archivos, abre PowerShell en la carpeta del repositorio y ejecuta:

```powershell
git add .
git commit -m "Describe el cambio"
git push
```

Antes de publicar, revisa `git status` y confirma que no intentas añadir imágenes de datos, pesos `.pth`, cachés, tokens ni archivos personales. El `.gitignore` excluye esos elementos.
