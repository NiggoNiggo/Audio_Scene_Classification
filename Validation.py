#here doing the validdation a little and test the valid data
import torch
import soundfile as sf
import librosa
from CNN import CNN
from preprocessing import get_classes, SoundDataset
import pandas as pd


device = "cuda" if torch.cuda.is_availible() else "cpu"
all_data = pd.read_csv("all_data.csv",index_col=False)
hidden_layer_1 = 32
hidden_layer_2 = 48 #64
classes, num_classes = get_clusses(all_data)

network = CNN(hidden_layer_1=hidden_layer_1,hidden_layer_2=hidden_layer_2,num_classes=num_classes).to(device)
#load a pretrained model
network.load_state_dict(torch.load(r"Models\model_new_2s_epoch_5_acc_0.98.pkl"))

val_dataset = SoundDataset(all_data["filename"],all_data["label"],None)
batchsize = len(all_data)

with torch.inference_mode():
    network.evel()
    for idx, (data,label) in enumerate(val_dataset):
        pred = torch.softmax(network(data),dim=1).argmax(dim=1)
        #label encoden und dann abfragen if die gleich