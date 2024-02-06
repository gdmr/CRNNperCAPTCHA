import torch
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ocrsolver import solve_captcha

# Carica il modello salvato

if torch.cuda.is_available():
    device = torch.device('cuda')
    loaded_model = torch.load('model.pth', map_location=device)
else:
    loaded_model = torch.load('model.pth', map_location=torch.device('cpu'))
ocr = OCR()  # Crea un'istanza OCR
ocr.crnn = loaded_model  # Sostituisci il modello CRNN con quello caricato

# Percorso dell'immagine da testare
img_path = '/path'

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
print(pred_text)