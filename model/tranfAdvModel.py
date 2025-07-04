import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader,Dataset
from torchvision import transforms

import numpy as np

# ---------------------------------------------------
# 1. MDTA y GDFN
# --------------------------------------------------- 
class MDTA(nn.Module):
    def __init__(self, embed_dim):
        """
        Módulo MDTA que opera directamente sobre tensores de forma (B, N, embed_dim).
        
        Args:
            embed_dim (int): Número de canales del embedding.
        """
        super(MDTA, self).__init__()
        # Proyectar la entrada para obtener concatenación de Q, K y V
        self.fc = nn.Linear(embed_dim, embed_dim * 3)
        # Capa de proyección final para la salida
        self.out = nn.Linear(embed_dim, embed_dim)
        # Parámetro de escalado learnable; se inicializa en 1.0
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor de forma (B, N, embed_dim)
            
        Returns:
            torch.Tensor: Salida de forma (B, N, embed_dim).
        """
        # Calcular Q, K, V a partir de la proyección lineal
        # x: (B, N, embed_dim) -> (B, N, 3 * embed_dim)
        qkv = self.fc(x)
        # Dividir en tres tensores a lo largo de la última dimensión
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Cada uno de forma (B, N, embed_dim)
        
        # Calcular la atención escalada: [B, N, N] = (Q * K^T) / sqrt(D)
        d = x.size(-1)
        attn = torch.bmm(q, k.transpose(1, 2)) / (d ** 0.5)
        # Aplicar la escala learnable y softmax
        attn = F.softmax(attn / self.scale, dim=-1)
        
        # Calcular la salida de la atención: (B, N, embed_dim)
        out = torch.bmm(attn, v)
        # Proyección final y retorno
        out = self.out(out)
        return out

class GDFN(nn.Module):
    def __init__(self, embed_dim, expansion=4):
        """
        Módulo GDFN que opera directamente sobre tensores de forma (B, N, embed_dim)
        utilizando una red feedforward con mecanismo de gating.
        
        Args:
            embed_dim (int): Número de canales del embedding.
            expansion (int or float): Factor de expansión para la capa oculta, según la fórmula:
                                      hidden_dim = embed_dim * expansion.
        """
        super(GDFN, self).__init__()
        hidden_dim = int(embed_dim * expansion)
        # Primera capa lineal que expande la dimensión
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # Segunda capa lineal que reduce nuevamente la dimensión
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        # Capa learnable para calcular la compuerta
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor de forma (B, N, embed_dim)
            
        Returns:
            torch.Tensor: Salida de forma (B, N, embed_dim)
        """
        # Guardar la entrada para la conexión residual
        residual = x
        # Primera rama: expandir y aplicar GELU
        x1 = self.fc1(x)
        x1 = self.act(x1)
        x1 = self.fc2(x1)  # Regresa a embed_dim
        # Cálculo de la compuerta
        gate = torch.sigmoid(self.gate(x))
        # La salida se modula con la compuerta y se retorna con conexión residual
        out = x1 * gate
        return out + residual

# ---------------------------------------------------
# 2. Bloque Transformer a usar en el generador y en el discriminador. 
# ---------------------------------------------------  

class GenTransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        """
        Bloque transformer para el generador que utiliza MDTA y GDFN,
        asumiendo que ambos módulos se han modificado para trabajar con tensores 3D.
        
        Args:
            embed_dim (int): Número de canales del embedding (la dimensión de cada token).
        """
        super(GenTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Ahora se supone que MDTA y GDFN aceptan tensores de forma (B, N, embed_dim)
        self.MDTA1 = MDTA(embed_dim)
        self.GDFN1 = GDFN(embed_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensores de entrada de forma (B, N, embed_dim),
                              donde N es el número de tokens.
        
        Returns:
            torch.Tensor: Salida del bloque transformer, de forma (B, N, embed_dim).
        """
        # Primera normalización de tokens
        x_norm = self.norm1(x)
        # Aplicar MDTA; se asume que devuelve un tensor de forma (B, N, embed_dim)
        mdta_out = self.MDTA1(x_norm)
        # Conexión residual
        x = x + mdta_out
        
        # Segunda normalización de tokens
        x_norm = self.norm2(x)
        # Aplicar GDFN; se asume que devuelve un tensor de forma (B, N, embed_dim)
        gdfn_out = self.GDFN1(x_norm)
        # Conexión residual
        x = x + gdfn_out
        
        return x
# ---------------------------------------------------
# 3. Módulos Encoder, Decoder y Refinador a usar en el generador
# ---------------------------------------------------  
class EncoderModule(nn.Module):
    def __init__(self, embed_dim, num_blocks):
        """
        Módulo Encoder que utiliza bloques Transformer personalizados (con MDTA y GDFN).
        
        Args:            
            embed_dim (int): Número de canales del embedding inicial.
            num_blocks (list of int): Número de bloques transformers en el módulo
        """
        super(EncoderModule, self).__init__()
        # Secuencia de bloques transformer
        self.tranformerBlocks = nn.Sequential(*[GenTransformerBlock(embed_dim) for _ in range(num_blocks)])
        # Capa para downsampling: usando una convolución con stride=2 para reducir la resolución espacial a la mitad.
        self.downsampler = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=2, padding=1)
        
    
    def forward(self, x):
        """
        Procesa el mapa de características y devuelve el resultado
        
        Args:
            x (torch.Tensor): Mapa de características de forma (B, embed_dim, H, W).
        
        Returns:
           output (torch.Tensor): Resultado downsampleado, forma (B, embed_dim, H, W).
        """
        
        B, C, H, W = x.shape
        # Aplanar las dimensiones espaciales para formar tokens: (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        # Procesar los tokens con los bloques transformer
        tokens = self.tranformerBlocks(tokens)
        # Reconstruir el mapa de características: (B, C, H, W)
        output = tokens.transpose(1, 2).view(B, C, H, W)
        
        return output
    
class DecoderModule(nn.Module):
    def __init__(self, embed_dim, num_blocks):
        """
        Módulo Decoder que utiliza bloques Transformer personalizados (con MDTA y GDFN).
        
        Args:            
            embed_dim (int): Número de canales del embedding inicial.
            num_blocks (list of int): Número de bloques transformers en el módulo
        """
        super(DecoderModule, self).__init__()
        # Secuencia de bloques transformer
        self.tranformerBlocks = nn.Sequential(*[GenTransformerBlock(embed_dim) for _ in range(num_blocks)])
        # Capa de upsampling: se utiliza para aumentar la resolución espacial.
        self.upsampler = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1)
        )

        
    
    def forward(self, x):
        """
        Procesa el mapa de características y devuelve el resultado
        
        Args:
            x (torch.Tensor): Mapa de características de forma (B, embed_dim, H, W).
        
        Returns:
           output (torch.Tensor): Resultado downsampleado, forma (B, embed_dim, H, W).
        """
        
        B, C, H, W = x.shape
        # Aplanar el mapa para formar una secuencia de tokens: (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        # Procesar la secuencia de tokens con los bloques transformer personalizados
        tokens = self.tranformerBlocks(tokens)
        # Reconstruir el mapa de características: (B, C, H, W)
        output = tokens.transpose(1, 2).view(B, C, H, W)
        
        return output

class RefinementModule(nn.Module):
    def __init__(self, embed_dim, num_blocks):
        """
        Módulo de refinamiento que procesa el mapa de características para mejorarlo
        sin cambiar su resolución o número de canales.
        
        Args:
            embed_dim (int): Dimensión (número de canales) de cada token.
            num_blocks (int): Número de bloques transformer a aplicar.
        """
        super(RefinementModule, self).__init__()
        # Crear una secuencia de bloques transformer personalizados
        self.blocks = nn.Sequential(*[GenTransformerBlock(embed_dim) for _ in range(num_blocks)])
        # Normalización final para estabilizar la salida
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Procesa el mapa de características y retorna un mapa refinado.
        
        Args:
            x (torch.Tensor): Mapa de características de forma (B, embed_dim, H, W).
            
        Returns:
            torch.Tensor: Mapa de características refinado, de la misma forma (B, embed_dim, H, W).
        """
        B, C, H, W = x.shape
        # Aplanar las dimensiones espaciales para formar una secuencia de tokens: (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        # Procesar la secuencia mediante los bloques transformer
        refined_tokens = self.blocks(tokens)
        # Reconstruir el mapa de características a partir de la secuencia
        refined_features = refined_tokens.transpose(1, 2).view(B, C, H, W)
        # Aplicar normalización: Permutamos para normalizar en la dimensión de los canales
        refined_features = self.norm(refined_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Añadir una conexión residual para conservar la información original
        output = x + refined_features
        return output

# ---------------------------------------------------
# 4. Generador
# ---------------------------------------------------
class SRTransG(nn.Module):
    def __init__(self, input_dim, embed_dim, token_dim, patch_size, num_levels, num_refinement, encoder_transformer_blocks, decoder_transformer_blocks, refinement_transformer_blocks):
        """
        Generador que utiliza división en parches.
        
        Args:
            imput_dim (int): Número de canales de las imagenes.
            embed_dim (int): Número de canales tras la primera convolución.
            patch_size (int): Tamaño del parche con dimensiones patch_size x patch_size
            token_dim (int): Dimensión a la que se proyectan los tokens (para reducir el uso de memoria). Solo aplica cuanndo se usan parches
            num_levels (int): Número de niveles en el encoder y decoder.
            num_refinement (int): Número de módulos de refinamiento tras el decoder.
            encoder_transformer_blocks (list of int): Lista con el número de bloques transformer en cada nivel del encoder.
            decoder_transformer_blocks (list of int): Lista con el número de bloques transformer en cada nivel del decoder.
            refinement_transformer_blocks (list of int): Lista con el número de bloques transformer para cada módulo de refinamiento.
        """
        super(SRTransG, self).__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_refinement = num_refinement
        self.patch_size = patch_size
        self.token_dim = token_dim        

        self.debug = False

        if len(encoder_transformer_blocks)!=num_levels:
            raise ValueError('Length of encoder_transformer_blocks parameter muest be equal to the value of num_levels parameter')
        if len(decoder_transformer_blocks)!=num_levels:
            raise ValueError('Length of decoder_transformer_blocks parameter muest be equal to the value of num_levels parameter')
        if len(refinement_transformer_blocks)!=num_refinement:
            raise ValueError('Length of refinement_transformer_blocks parameter muest be equal to the value of num_refinement parameter')
        
        
        self.decoderModules = nn.ModuleList()
        self.encoderModules = nn.ModuleList()
        self.refinementModules = nn.ModuleList()
        self.downsamplers = nn.ModuleDict()
        self.upsamplers = nn.ModuleDict()
        self.channelRedctn = nn.ModuleDict()
        self.featDecDict = dict()

        self.simpleUpsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
                    nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1))
        
        # Capa inicial para extraer características de la imagen LR
        self.conv_in = nn.Conv2d(input_dim, embed_dim, kernel_size=3, padding=1)

        if patch_size>0: #Solo si se usan parches            
            level_dim=token_dim
            token_in_dim_1=embed_dim*patch_size*patch_size # aplica para cuando el tensor tiene dimensión (B,H,W,C)
            token_in_dim_2=(embed_dim//2)*patch_size*patch_size # aplica para cuando el tensor tiene dimensión (B,H*2,W*2,C/2)
            self.token_proj = nn.Linear(token_in_dim_1, token_dim)
            self.token_reproj = nn.Linear(token_dim//2, token_in_dim_2)                    
        else:
            level_dim=embed_dim
        for i in np.arange(0,num_levels):            
            self.decoderModules.append(DecoderModule(level_dim,encoder_transformer_blocks[i]))
            self.encoderModules.append(EncoderModule(level_dim // 2,encoder_transformer_blocks[i]))
            self.downsamplers[str(i)] = nn.Conv2d(level_dim, level_dim * 2, kernel_size=3, stride=2, padding=1)
            self.upsamplers[str(i)] = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
                    nn.Conv2d(level_dim, level_dim // 2, kernel_size=3, padding=1))
            self.channelRedctn[str(i)] = nn.Conv2d(level_dim, level_dim // 2, kernel_size=3, padding=1)
            level_dim=level_dim*2            

        if patch_size>0:
            for i in np.arange(0,num_refinement):                      
                self.refinementModules.append(RefinementModule(token_dim//2,refinement_transformer_blocks[i]))            
        else:
            for i in np.arange(0,num_refinement):            
                self.refinementModules.append(RefinementModule(embed_dim // 2,refinement_transformer_blocks[i]))            
        
        # Capa final para reconstruir la imagen SR a partir de las características procesadas
        self.conv_out = nn.Conv2d(embed_dim // 2, input_dim, kernel_size=3, padding=1)

    def patchify_features(self, feat, token_proj):
        """
        Separa las características en parches sin promediarlos y 
        retorna un tensor de forma (B, token_dim, Hp, Wp),
        donde:
        - B es el tamaño del batch,        
        - H y W son las dimensiones espaciales,
        - Hp = H // patch_size y Wp = W // patch_size.
        
        Args:
            feat (torch.Tensor): Tensor de forma (B, C, H, W).            
        
        Returns:
            torch.Tensor: Tensor de forma (B, token_dim, Hp, Wp).
        """
        # feat tiene forma (B, C, H, W)
        B, C, H, W = feat.shape
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        # Usamos nn.Unfold para extraer parches
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)    
        patches = unfold(feat)  # (B, C*patch_size*patch_size, L) donde L = Hp*Wp
        patches = patches.transpose(1, 2)  # (B, L, C*patch_size*patch_size)        
        token = token_proj(patches) # (B, L, token_dim)
        patches = token.transpose(1, 2)  #  (B, token_dim, L)
        # Reorganizamos la dimensión L en dos dimensiones espaciales
        patches = patches.view(B, self.token_dim, Hp, Wp)# (B, token_dim, Hp, Wp)
        return patches

    def unpatchify_features(self, patches, output_size,token_reproj):
        """
        Reconstruye el tensor original a partir de los parches.
        
        Args:
            patches (torch.Tensor): Tensor de forma (B, token_dim, Hp, Wp),
                                    donde Hp = H // patch_size y Wp = W // patch_size.
            
            output_size (tuple): Dimensiones (H, W) deseadas, que deben ser Hp*patch_size y Wp*patch_size.
        
        Returns:
            torch.Tensor: Tensor reconstruido de forma (B, C, H, W).
        """
        B, D, Hp, Wp = patches.shape  # D = token_dim
        # Primero, reorganizamos a la forma (B, D, L)
        L = Hp * Wp
        patches_reshaped = patches.view(B, D, L) #(B,D,L)
        patches_reshaped = patches_reshaped.transpose(1, 2) # (B,L,D)
        token = token_reproj(patches_reshaped) #(B,L,C*patch_size*patch_size)
        token = token.transpose(1, 2) #(B, C*patch_size*patch_size,L)

        # Ahora usamos nn.Fold para reconstruir el tensor original.
        fold = nn.Fold(output_size=output_size, kernel_size=self.patch_size, stride=self.patch_size)
        feat = fold(token)  # (B, C, H, W)
        return feat
        
    def forward(self, x):
        # x: Imagen LR de forma (B, 3, H, W) donde H y W son dimensiones LR (por ejemplo, 64x64)
        feat = self.conv_in(x)                # (B, embed_dim, H, W)

        feat_up = self.simpleUpsample(feat)

        if self.patch_size>0:
            # Dividir en parches justo después de conv_in:
            patches = self.patchify_features(feat,self.token_proj)  # (B, Hp*Wp,embed_dim*patch_size*patch_size), donde Hp=H/patch_size y Wp=W/patch_size           

            # Extraemos las carácterísticas pasando el resultado de la capa convolucional inicial, dividida en parches, por el flujo de modulos decodificadores
            featDecTemp = patches
        else:
            featDecTemp = feat

        level_dim=self.embed_dim  
        #Pipeline del decoder
        self.featDecDict.clear()
        if self.debug: print("Pipeline Decoder")
        for i in np.arange(0,self.num_levels):
            if self.debug: print(f"Level {i}")
            featDec=self.decoderModules[i](featDecTemp)            
            self.featDecDict[i] = featDec
            featDecTemp = self.downsamplers[str(i)](featDec)
            if i<(self.num_levels-1):
                level_dim=level_dim*2  #porque en la arquitectura el ultimo decoder no hace downnsampling

        #pasamos las características decodificadas por el flujo de codificadores
        featEncTemp = None
        #Pipeline del encoder
        if self.debug: print("Pipeline Encoder")
        for i in np.arange(0,self.num_levels)[::-1]: 
            if self.debug: print(f"Level {i}")
            featEnc1 = self.featDecDict[i]
            featEnc2 = None
            if  i>0: #para evitar que en el nivel 0 quiera obtener el resultado del nivel -1
                featEnc2 = self.featDecDict[i-1]
                featInTemp = self.upsamplers[str(i)](featEnc1)
                featInTemp = torch.cat((featInTemp,featEnc2),1)
                featInTemp = self.channelRedctn[str(i)](featInTemp)
            else:
                featInTemp = self.upsamplers[str(i)](featEnc1)            
            
            if i < (self.num_levels-1): #dado que para  ultimo nivel el input del decoder es un derviado del resultado del encoder del último nivel y del resultado del encoder del niverl inmediatamente superior
                if True:#not(i==0 and self.patch_size>0):
                    featEncTemp = self.upsamplers[str(i)](featEncTemp)
                    if self.debug: print("featEncTemp upsampled")
                featInTemp = torch.cat((featInTemp,featEncTemp),1)
                featInTemp = self.channelRedctn[str(i)](featInTemp)                            

            featEncTemp = self.encoderModules[i](featInTemp)
            level_dim = level_dim // 2
        
        featRefTemp = featEncTemp
        if self.debug: print("Pipeline Refinement")
        for i in np.arange(0,self.num_refinement): 
            featRefTemp = self.refinementModules[i](featRefTemp)          

        
        if self.patch_size>0:
            output_size = (feat_up.shape[2], feat_up.shape[3])

            feat_recon = self.unpatchify_features(featRefTemp, output_size,self.token_reproj) 

            featFinal = feat_recon+feat_up
        else:
            featFinal = featRefTemp+feat_up
        out = self.conv_out(featFinal)       # (B, 3, H, W)

        return out