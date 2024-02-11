import torch
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



def solve_captcha(model_path, img_path):

    # Carica il modello salvato
    loaded_model = torch.load(model_path, map_location=torch.device('cpu'))

    # Crea un'istanza OCR e carica il modello
    ocr = OCR()
    ocr.crnn = loaded_model

    # Carica e trasforma l'immagine
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img)

    # Aggiungi una dimensione batch
    img_tensor = img_tensor.unsqueeze(0)

    # Esegui la predizione
    logits = ocr.predict(img_tensor)

    # Decodifica la predizione
    pred_text = ocr.decode(logits.cpu())
    
    return pred_text


