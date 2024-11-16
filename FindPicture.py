from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
from PIL import Image
import torch
import os
import tkinter as tk
from tkinter import filedialog


#model creatin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm")
model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-large-itm-mlm").to(device)


def add_image_scores(scores, directory, description):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            try:
                image = Image.open(f).resize((256,256))
                #similarity calculation
                encoding = processor(image, description, return_tensors="pt").to(device)
                results = model(**encoding).logits
                outputString = 'Image {}, description "{}", similarity {}'.format(f, description, int(100 *results[0][1].item())/100)    
                print(outputString)
                scores[-results[0][1].item()] = f
            except Exception:
                print("Can not open ", f)
                continue
        #elif os.path.isdir(f):
            #add_image_scores(scores, f, description) 
    return   


folder_name = filedialog.askdirectory(title="Select a Directory")
description ="men wearing a helmet"
scores = dict()
add_image_scores(scores, folder_name, description)
myKeys = list(scores.keys())
myKeys.sort()

for i in range(3):
    Image.open(scores[myKeys[i]]).show()
    print(scores[myKeys[i]], myKeys[i])
