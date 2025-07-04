import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import TensorDataset
from PIL import Image
import tifffile as tiff
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle


def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    """
    Asegura que el arreglo tenga forma (C, H, W).
    - Si viene como (H, W), agrega canal único.
    - Si viene como (H, W, C), transpone.
    - Si ya es (C, H, W), no hace nada.
    """
    if arr.ndim == 2:
        # Mono-banda → 1 x H x W
        return arr[np.newaxis, ...]
    elif arr.ndim == 3:
        h, w, c = arr.shape
        if c not in (1,):  # típico caso H, W, C multibanda
            return arr.transpose(2, 0, 1)
        else:
            # H, W, 1 → 1, H, W
            return arr[:, :, 0][np.newaxis, ...]
    else:
        raise ValueError(f"Formato de array inesperado: ndim={arr.ndim}")

def normalize(arr): #probar con standarización / normalización por z-score
    arr = arr.astype('float32')
    if arr.dtype == np.float32:
        # si ya has hecho la conversión a float, 
        # puedes asumir que vino de uintX y divido abajo
        pass
    # supongamos que la matriz original era uint8 o uint12
    # recuperamos el max según el dtype original                
    max_val = 4095 if arr.max() > 255 else 255
    bitdepth = 12 if arr.max() > 255 else 8
    return arr / max_val , bitdepth

def make_dataset(hr_dir, lr_dir, hr_list, lr_list,suffix,im_Extensions):
    hr_tensors, lr_tensors = [], []
    
    if lr_list is None:       
        lr_list = [
            f"{name}{suffix}{ext}"
            for name, ext in map(os.path.splitext, hr_list)
        ]
        
    bitdepth = 0 #información a usar para luego generar las imágenes en el proceso de inferencia
    
    for hr_fn, lr_fn in zip(hr_list, lr_list):
        ext = os.path.splitext(hr_fn)[1].lower()
        hr_path = os.path.join(hr_dir, hr_fn)
        lr_path = os.path.join(lr_dir, lr_fn)
        if ext in [".tif", ".tiff"]:
            # Leer con tifffile
            hr_arr = tiff.imread(os.path.join(hr_dir, hr_fn))
            lr_arr = tiff.imread(os.path.join(lr_dir, lr_fn))
        else:
            hr_arr = np.array(Image.open(hr_path).convert("RGB"))
            lr_arr = np.array(Image.open(lr_path).convert("RGB"))
            # Asegurar forma (C, H, W)
        hr_arr = _ensure_chw(hr_arr)
        lr_arr = _ensure_chw(lr_arr)

        
        hr_norm, tbitdepth = normalize(hr_arr)
        lr_norm, _ = normalize(lr_arr)
        bitdepth = max(bitdepth, tbitdepth)
    
        hr_tensors.append(torch.from_numpy(hr_norm))
        lr_tensors.append(torch.from_numpy(lr_norm))

    # Devuelve (LR_batch, HR_batch) como antes
    return torch.stack(lr_tensors), torch.stack(hr_tensors),bitdepth


def load_dataset(
    hr_dir: str,
    lr_dir: str,
    suffix: str,
    hr_size: tuple,
    max_images: int = 100,
    val_frac: float = 0.1,
    pickle_path: str = None,
    flag_only_pckl: bool= False,
    im_Extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
):
    """
    Carga y valida imágenes HR/LR, y las separa en train/val. En caso se pase un pickle_path reconstruye los DataLoaders de entrenamiento y validación a partir
    de un pickle con las listas de splits, y devuelve además esas listas.

    Args:
        hr_dir (str): Carpeta con imágenes HR.
        lr_dir (str): Carpeta con imágenes LR.
        suffix (str): Sufijo que distingue LR de HR (ej. 'x2').
        hr_size (tuple): (alto, ancho) esperado en HR.
        max_images (int): Si >0, número máximo de pares a usar. Si =0, usar todas. No aplica si se pasa un pickle_path.
        val_frac (float): Fracción [0,1) de imágenes para validación. No aplica si se pasa un pickle_path.
        pickle_path (str): Ruta al pickle con {'train': [...], 'val': [...]}.
        flag_only_pckl (bool): Indica que si en caso no se encuentre el pickle con los splits generará los splits desde cero.

    Returns:
        train_dataset (TensorDataset), val_dataset (TensorDataset),
        train_names (list[str]), val_names (list[str])
    """
    train_hr = None
    val_hr = None
    train_lr = None
    val_lr = None
        
    if pickle_path is not None:
        # 1) Cargamos las listas de nombres desde el pickle
        path = Path(pickle_path)
        if path.exists(): # 2)validamos si existe el archivo pickle
            # 3)cargamos el split
            with path.open("rb") as f:
                split = pickle.load(f)
                train_hr = split["train"]
                val_hr   = split["val"]
        elif flag_only_pckl:            
            raise Exception("Pickle file does not exist.")            
    else:                
        # 1) Listamos y filtramos todas las HR que tienen EXACTAMENTE hr_size:
        all_hr = [f for f in sorted(os.listdir(hr_dir))
                  if f.lower().endswith(im_Extensions)]
        valid_hr = []
        for fn in all_hr:
            with Image.open(os.path.join(hr_dir, fn)) as im:
                if im.size == (hr_size[1], hr_size[0]):
                    valid_hr.append(fn)
        total_hr = len(valid_hr)
    
        # 2) Chequeo de cantidad:
        if max_images > 0 and total_hr < max_images:
            raise RuntimeError(
                f"Sólo hay {total_hr} imágenes HR con resolución {hr_size}, "
                f"pero se pidió max_images={max_images}."
            )
        if max_images == 0:
            max_images = total_hr
    
        # 3) Seleccionamos aleatoriamente max_images nombres de HR:
        selected_hr = valid_hr[:max_images]
        
        # 4) Validamos que exista LR con sufijo y tamaño = hr_size//2:
        lr_names = []
        for hr_fn in selected_hr:
            base, ext = os.path.splitext(hr_fn)
            lr_fn = base + suffix + ext
            lr_path = os.path.join(lr_dir, lr_fn)
            if not os.path.exists(lr_path):
                raise RuntimeError(f"No existe LR para '{hr_fn}' → buscado '{lr_fn}'.")
            with Image.open(lr_path) as im:
                expected = (hr_size[1]//2, hr_size[0]//2)
                
                if im.size != expected:
                    raise RuntimeError(
                        f"LR '{lr_fn}' tiene tamaño {im.size}, "
                        f"pero esperaba {expected}."
                    )
            lr_names.append(lr_fn)
    
        # 5) Dividimos nombres en train / val:
        train_hr, val_hr, train_lr, val_lr = train_test_split(
            selected_hr, lr_names,
            test_size=val_frac, random_state=42
        )

    # 6) Transform a tensor (sin resize, ya están en tamaño correcto)        
    lr_train, hr_train, bitdepth = make_dataset(hr_dir, lr_dir, train_hr, train_lr, suffix, im_Extensions)
    lr_val, hr_val, _   = make_dataset(hr_dir, lr_dir, val_hr, val_lr, suffix, im_Extensions)

    train_dataset = TensorDataset(lr_train, hr_train)
    val_dataset   = TensorDataset(lr_val,   hr_val)

    return train_dataset, val_dataset, train_hr, val_hr, bitdepth