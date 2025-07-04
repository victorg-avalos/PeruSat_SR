import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------
# 1.  Discriminador ViT
# ---------------------------------------------------

class DiscTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        """
        Bloque Transformer personalizado para el discriminador.
        
        Args:
            embed_dim (int): Número de canales del embedding.
            num_heads (int): Número de cabezas en la atención múltiple.
            mlp_ratio (int): Multiplicador para definir la dimensión de la capa MLP, 
                             de acuerdo con: embed_dim * mlp_ratio.
        """
        super(DiscTransformerBlock, self).__init__()
        # Normalización de entrada
        self.norm1 = nn.LayerNorm(embed_dim)
        # Atención múltiple
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Segunda normalización para la salida de la atención
        self.norm2 = nn.LayerNorm(embed_dim)
        # Capa MLP: la entrada se expande a embed_dim * mlp_ratio y luego se reduce a embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
        
    def forward(self, x):
        # x de forma (B, N, embed_dim), donde N es el número de tokens.
        # Aplicar la primera normalización.
        x_norm = self.norm1(x)
        # Aplicar atención múltiple: se usan la misma secuencia para consulta, clave y valor.
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        # Conexión residual con la salida de la atención.
        x = x + attn_out
        # Aplicar la segunda normalización.
        x_norm = self.norm2(x)
        # Procesar con el MLP.
        mlp_out = self.mlp(x_norm)
        # Conexión residual con la salida del MLP.
        x = x + mlp_out
        return x



class ViTDiscriminator(nn.Module):
    def __init__(self, hr_size, in_channels=8, patch_size=16, embed_dim=64, depth=6, num_heads=4, mlp_ratio=4, num_classes=1):
        """
        Discriminador basado en Vision Transformer.
        
        Este módulo toma la imagen de entrada (por ejemplo, 6 canales), la divide en parches
        mediante un Conv2d, añade un token de clase learnable y embeddings posicionales, y procesa
        los tokens a través de varios bloques transformer (DiscTransformerBlock). Finalmente, se extrae
        el token de clase (tras normalización) para producir una probabilidad de que la imagen sea real.
        
        Args:
            hr_size (tuple(int)): Dimensiones de la imagen de alta resoluciónn 
            in_channels (int): Número de canales de la imagen de entrada.
            patch_size (int): Tamaño del parche cuadrado.
            embed_dim (int): Número de canales del embedding de cada parche.
            depth (int): Número de bloques transformer (DiscTransformerBlock) a utilizar.
            num_heads (int): Número de cabezas en la atención múltiple.
            mlp_ratio (int): Multiplicador para la dimensión de la capa MLP en cada bloque.
            num_classes (int): Número de clases en la salida (para clasificación binaria, 1).
        """
        super(ViTDiscriminator, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding: se utiliza una convolución para extraer parches con stride igual al tamaño del parche
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Se asume una dimensión de entrada fija (por ejemplo, 128x128). Calculamos el número de parches:
        num_patches = (hr_size[0] // patch_size) * (hr_size[1] // patch_size)
        # Token de clase learnable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Embedding posicional para cada token, incluyendo el token de clase.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Crear una lista de bloques transformer para el discriminador
        self.blocks = nn.ModuleList([
            DiscTransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        
        # Capa de normalización final
        self.norm = nn.LayerNorm(embed_dim)
        # Cabeza de clasificación: mapea el token de clase a la salida deseada (por ejemplo, 1)
        self.head = nn.Linear(embed_dim, num_classes)
        self.sigmoid = nn.Sigmoid()  # Convierte la salida a probabilidad
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x):
        # x: Imagen de entrada de forma (B, in_channels, H, W)
        B, _, H, W = x.shape
        # Aplicar patch embedding para convertir la imagen en parches embebidos.
        x = self.patch_embed(x)  # Forma resultante: (B, embed_dim, H/patch_size, W/patch_size)
        # Aplanar las dimensiones espaciales y transponer: (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Prepend the learnable class token.
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, num_patches+1, embed_dim)
        
        # Añadir embeddings posicionales.
        x = x + self.pos_embed
        
        # Procesar la secuencia con los bloques transformer.
        for block in self.blocks:
            x = block(x)  # Cada bloque mantiene la forma (B, num_patches+1, embed_dim)
        
        # Normalización final.
        x = self.norm(x)
        # Usar el token de clase para la clasificación.
        cls_out = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_out)  # (B, num_classes)
        prob = self.sigmoid(logits)   # Convertir logits a probabilidad (0 a 1)
        return prob