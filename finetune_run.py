
#%%
import json
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
from torchvision import transforms
import os
import cv2
import torch.nn.functional as F

#%%
# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu" 

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

#%%
raw_df = pd.read_csv('./CC3M_dataset/Image_Labels_Subset_Train_GCC-Labels-training.tsv', sep='\t', names=['caption', 'link', 'label', 'mola', 'ci'])
json_data = "./generate_intent_gpt3.5/intent_full_final.json"
with open(json_data, "r", encoding='UTF8') as f:
    intent = json.load(f)
    
id_word = []
for i in intent:
    # extract key and value
    for key, value in i.items():
        if key.startswith("caption"):
            caption_id = key.replace("caption", "")
            id_word.append((int(caption_id), i["intent_word"]))

id_word = pd.DataFrame(id_word, columns=['id', 'abstract_label']) 
# id_word.set_index('id', inplace=True)    

df = pd.merge(raw_df, id_word, left_index=True, right_index=True)
df['label'] = df['label'].str.split(',')
df.dropna(inplace=True)
df = df[["id", "label", "abstract_label"]]
df["total_label"] = df["label"] + df["abstract_label"]
df_expanded = df.explode("total_label")

#%%
ids = df_expanded["id"].unique()
train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42) # You can adjust the test_size as needed
train_df = df_expanded[df_expanded["id"].isin(train_ids)]
test_df = df_expanded[df_expanded["id"].isin(test_ids)]

#%%
list_image_path = []
list_txt = []
for i in train_ids:
    img_path = './conceptual_img/image_' + str(i) + '.jpg'
    if os.path.exists(img_path):
        img = np.array(Image.open(img_path))
        if len(img.shape) == 3:
            selected_row = train_df[train_df['id'] == int(i)]
            text = selected_row['total_label'].values[0]
            for text in selected_row['total_label'].values:            
                list_image_path.append(img_path)
                list_txt.append(text)

#%%
# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

if device == "cpu":
  model.float()

# Prepare the optimizer
#optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

#%%

# Define a custom dataset
class image_title_dataset():
    def __init__(self, list_image_path, list_txt):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        #self.title  = clip.tokenize(list_txt)
        #self.title = [clip.tokenize(text) for text_list in list_txt for text in text_list]
        self.title = list_txt
    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        #print(idx)
        if idx < 0 or idx >= len(self.title):
            raise IndexError(f"Index {idx} is out of range for the dataset.")
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title
    
# Now you can create separate CustomDatasets for the train and test sets

list_txt_token = []
for text_list in list_txt:
    #list_txt_token.append(clip.tokenize(text_list)[0])
    list_txt_token.append(clip.tokenize(text_list))

train_dataset = image_title_dataset(list_image_path, list_txt_token)
#test_dataset = image_title_dataset(test_df, image_transform)

#%%
train_data_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
#test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)  # No need to shuffle the test data

#%%
temperature = 0.07

# Train the model
num_epochs = 5
losses = []

for epoch in range(num_epochs):
    pbar = tqdm(train_data_loader, total=len(train_data_loader))
    for batch in pbar:
        epoch_losses = []
        optimizer.zero_grad()

        images, texts = batch     
        texts = texts.squeeze()    
        images = images.to(device)
        texts = texts.to(device)
    
        logits_per_image, logits_per_text = model(images, texts)     

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        # Backward pass
        loss.backward()
        #optimizer.step()
        if device == "cpu":
            optimizer.step()
        else:
            # Convert model's parameters to FP32 format, update, and convert back
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        epoch_losses.append(loss.item())
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")
    
    mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(mean_epoch_loss)
    #model._save_to_state_dict("my_clip_model_v3_batch100_epoch_" + str(epoch+1) + ".pt")
    
    torch.save(model.state_dict(), 'clip_v3_batch100_epoch_' + str(epoch+1) + '.pt')

print("Finished Training")
#%%
#model._save_to_state_dict("my_clip_model_v3_batch10_epoch10.pt")
#torch.save(model.state_dict(), 'clip_v3_batch10_epoch10.pt')
#print("saved!")
#%%
print(f"losses : {losses}")

#%%

plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()
plt.savefig('finetune_batch10_epoch10.png')
# %%

