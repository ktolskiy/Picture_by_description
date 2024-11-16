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

serchSubfolders = False
description =""
count =3

root=tk.Tk()
root.wait_visibility()
# declaring string variable
# for description and pictures count
description_var = tk.StringVar()
count_var = tk.StringVar()
var = tk.IntVar()

def submit():
    global description
    global count
    global root
    description=description_var.get()
    count= count_var.get()
    root.quit()

def on_button_toggle():
    global serchSubfolders
    if var.get() == 1:
        serchSubfolders = True
    else:
        serchSubfolders = False
    

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
        elif os.path.isdir(f) and serchSubfolders:
            add_image_scores(scores, f, description) 
    return   


folder_name = filedialog.askdirectory(title="Select a Directory")

# creating a label for 
# request using widget Label
descr_label = tk.Label(root, text = 'Image description', font=('calibre',12, 'bold'))
 
# creating a entry for input
# request using widget Entry
descr_entry = tk.Entry(root,textvariable = description_var, font=('calibre',12,'normal'))
 
# creating a label for the returned pictures count
count_label = tk.Label(root, text = 'How many pictures to show', font = ('calibre',12,'bold'))
 
# creating a entry for the returned pictures count
count_entry=tk.Entry(root, textvariable = count_var, font = ('calibre',12,'normal'))

# Creating a Checkbutton
var = tk.IntVar()
checkbutton = tk.Checkbutton(root, text="Search Subdirectories", variable=var, 
                             onvalue=1, offvalue=0, command=on_button_toggle)

 
# creating a button using the widget 
# Button that will call the submit function 
sub_btn=tk.Button(root,text = 'Submit', command = submit)
 
# placing the label and entry in
# the required position using grid
# method
descr_label.grid(row=0,column=0)
descr_entry.grid(row=0,column=1)
count_label.grid(row=1,column=0)
count_entry.grid(row=1,column=1)
checkbutton .grid(row=2,column=0)
sub_btn.grid(row=3,column=1)
# performing an infinite loop 
# for the window to display
root.mainloop()

scores = dict()
add_image_scores(scores, folder_name, description)
myKeys = list(scores.keys())
myKeys.sort()
m = min(int(count), len(myKeys))

for i in range(m):
    Image.open(scores[myKeys[i]]).show()
    print(scores[myKeys[i]], myKeys[i])

