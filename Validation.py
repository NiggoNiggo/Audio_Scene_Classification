#here doing the validdation a little and test the valid data
import torch
import soundfile as sf
import librosa
from CNN import CNN
from pre_processing import get_classes,  SoundDataSet, split_data
from torch.utils.data import DataLoader
from torchmetrics import Accuracy,ConfusionMatrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import os 

def load_test_data(path):
    classes = os.listdir(path)
    test_data = pd.DataFrame(columns=["filename", "label"])
    for label in classes:
        all_files = librosa.util.find_files(os.path.join(path, label), ext="wav")
        temp_df = pd.DataFrame({'filename': [file for file in all_files], 'label': label})
        test_data = pd.concat([test_data,temp_df], ignore_index=True)
    encoder = LabelEncoder()
    test_data["label"] = encoder.fit_transform(test_data["label"])
    test_data.to_csv('validation_data.csv', index=False)

if __name__ == "__main__":

    load_test_data(r"C:\Users\analf\Desktop\Studium\Learn_NN\Datasets\Data\Test_data")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_data = pd.read_csv("validation_data.csv",index_col=False)
    hidden_layer_1 = 32
    hidden_layer_2 = 48 
    classes, num_classes = get_classes(all_data)

    network = CNN(hidden_layer_1=hidden_layer_1,hidden_layer_2=hidden_layer_2,num_classes=15).to(device)
    # load a pretrained model
    network.load_state_dict(torch.load(r"Models\model_new_2s_epoch_0_acc_0.97.pkl"))
    accuracy = Accuracy(task="multiclass",num_classes=15).to(device)
    con_mat = ConfusionMatrix(task="multiclass",num_classes=15).to(device)


    val_dataset = SoundDataSet(all_data["filename"],all_data["label"],None)
    val_loader = DataLoader(val_dataset,shuffle=True,num_workers=6)
    batchsize = len(all_data)
    print(batchsize)

    with torch.inference_mode():
        network.eval()
        preds = []
        labels = []
        for idx, (data,label) in enumerate(val_loader):
            data, label = data.to(device), label.to(device)
            y_pred = network(data)
            pred = torch.softmax(y_pred,dim=1).argmax(dim=1)
            preds.append(pred)
            labels.append(label)
        acc = Accuracy(torch.tensor(labels).to(device),torch.tensor(preds).to(device))
        print(acc)
        con_mat(torch.tensor(preds).to(device),torch.tensor(labels).to(device))
        con_mat.plot()
        plt.show()
            # label encoden und dann abfragen if die gleich
            