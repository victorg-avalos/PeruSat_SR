# PeruSat_SR
Repositorio para el código python usado en el entrenamiento de un modelo Transformer-Generativo-Advesarial para la generación de imagenes de superresolución a partir de un dataset de imagenes satelitales obtenidas del satelite PeruSat 2.

En este repositorio se han incluido lo siguiente:
- Carpeta modelos: Contiene el codigo fuente del modelo transformer y ViT implementado. El primero es usado como generador y el segundo como discriminador. Adicionalmente, esta el código fuente para la carga de dataset.
- Carpeta metricas: Contiene el código fuente para obtener las metricas PSNR y SSIM, asi como las funciones de loss utilizadas.
- Bicubic.ipynb : Código para generar imagenes de superresolución usando interpolación bicúbica
- CNN_Basic.ipynb: CNN básica usada para generar imagenes de superresolución
- Train_RGB.ipynb: Entrenamiento del modelo transformer puro
- Metrics_RGB.ipynb: Obtención de metricas del modelo transformer puro

