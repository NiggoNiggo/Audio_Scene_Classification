#here doing the validdation a little and test the valid data
import torch
import soundfile as sf
import librosa,random
from CNN import CNN
from pre_processing import get_classes,  SoundDataSet, split_data
from torch.utils.data import DataLoader
from torchmetrics import Accuracy,ConfusionMatrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import os 
# from Kompressor import Kompressor





#0 = cafe, 1 = Music, 2 = Street

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
    test_path = r"C:\Users\analf\Desktop\Studium\Learn_NN\Datasets\Data\Test_data"
    load_test_data(test_path)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_data = pd.read_csv("validation_data.csv",index_col=False)
    hidden_layer_1 = 32
    hidden_layer_2 = 48 
    classes, num_classes = get_classes(all_data)
    len_cafe = len(os.listdir(os.path.join(test_path,"cafe")))
    len_music = len(os.listdir(os.path.join(test_path,"music")))
    len_street = len(os.listdir(os.path.join(test_path,"street")))
    print(len_cafe,len_music,len_street)
    

    network = CNN(hidden_layer_1=hidden_layer_1,hidden_layer_2=hidden_layer_2,num_classes=16).to(device)
    # load a pretrained model
    network.load_state_dict(torch.load(r"Models\test_new_length_10_acc_0.93.pkl"))
    accuracy = Accuracy(task="multiclass",num_classes=16).to(device)
    con_mat = ConfusionMatrix(task="multiclass",num_classes=16).to(device)


    val_dataset = SoundDataSet(all_data["filename"],all_data["label"],None,mode="val")
    val_loader = DataLoader(val_dataset,shuffle=True,num_workers=6)
    batchsize = len(all_data) #hier noch andere Zahlen ausporbieren

    pred_class1 = []
    pred_class2 = []
    pred_class3 = []
    
    with torch.inference_mode():
        network.eval()
        preds = []
        labels = []
        for idx, (data,label) in enumerate(val_loader):
            data, label = data.to(device), label.to(device)
            y_pred = network(data)
            pred = torch.softmax(y_pred,dim=1).argmax(dim=1)
            if pred.item() in [2,6,7,11,8,9] and label == 0:
                pred_class1.append(pred.item())
            elif pred.item() == 10 and label == 1:
                pred_class2.append(pred.item())
            elif pred.item() in [0,1,3,4,5,12,13,14,15] and label == 2:
                pred_class3.append(pred.item())

            #contained labels with the rigth order:
            
#             ['beach' 'bus' 'cafe/restaurant' 'car' 'city_center' 'forest_path'
#               'grocery_store' 'home' 'library' 'metro_station' 'music' office' 'park'
#               'residential_area' 'train' 'tram']
            
            preds.append(pred)
            labels.append(label)
        con_mat(torch.tensor(preds).to(device),torch.tensor(labels).to(device))
        con_mat.plot()
        plt.show()
    
    all_correct = len(pred_class3)+len(pred_class1)+len(pred_class2)
    acc = round(all_correct/batchsize,2)*100
    print(f"len_cafe:{len_cafe}, len_cafe_pred:{len(pred_class1)},lwn_street:{len_street}, len_street_pred: {len(pred_class3)}")
    print(f"len music: {len_music}, len_music_pred: {len(pred_class2)}")
    len_cafe = len(pred_class1)/len_cafe#cafe
    len_street = len(pred_class3)/len_street#street
    len_music = len(pred_class2)/len_music
    print(f"acc class cafe: {round(len_cafe,2)*100}, acc class street: {round(len_street,2)*100}",
          f"acc class music: {round(len_music,2)*100}\noverall acc: {acc}")

    #vlt nochmal ne andere batchsize ausprbieren könnte auch noch funktionieren
    
    
    #take a prediction and change the signalprocessing frim the compressor
    
    prediction = 0
    
    if prediction == 0:
        print("match signalprocessing to class cafe")
        filter_gain = [-50,10,-50]
        treshold_compressor = 0.8 #bzw muss das als Pegel angegeben werdne
        threshold_expander = 0.4 #hier wie ooben
        ratio_compressor = 0.5
        ratio_expander = 4
        #weiter größen später
    elif prediction == 1:
        print("match signalprocessign to class music")
        filter_gain = [1,1,1] #hier schauen das alles aus ist für natürlichen Klang
        treshold_compressor = 0 #bzw muss das als Pegel angegeben werdne
        threshold_expander = 1 #hier wie ooben
        ratio_compressor = 1
        ratio_expander = 1
    elif prediction == 2:
        print("match signalprocessing ro class street")
        filter_gain = [-100,-50,-50]
        treshold_compressor = 0.8 #bzw muss das als Pegel angegeben werdne
        threshold_expander = 0 #hier wie ooben
        ratio_compressor = 0.2
        ratio_expander = 1
    # kompressor = Kompressor()
    #soll da nun was poassieren? oder nur die parameter geändert wereden?