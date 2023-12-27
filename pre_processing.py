import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch, librosa, os

#---------paths: 
#path to the csv file with all the label and filename information
data_path = r"C:\Users\analf\Desktop\Studium\Learn_NN\Datasets\Data"
info_segmented = r"C:\Users\analf\Desktop\Studium\Learn_NN\Datasets\Data\segmented_audio.csv" #just segmented audio
info_audio_1 = r"C:\Users\analf\Desktop\Studium\Learn_NN\Datasets\Data\meta.txt"
info_audio_2 = r"C:\Users\analf\Desktop\Studium\Learn_NN\Datasets\Data\meta1.txt"
music_path = r"C:\Users\analf\Desktop\Studium\Learn_NN\Datasets\Data\music"
#type in here above your paths


#-----------global variables



#function to splitt val and train data 
def split_data(df:pd.DataFrame):
    #split data
    X_train, X_test, y_train, y_test = train_test_split(df["filename"].to_numpy(),df["label"].to_numpy(),test_size=0.4,shuffle=True)
    X_test, X_val, y_test,y_val = train_test_split(X_test,y_test,test_size=0.5)
    return X_train, X_test, X_val, y_train, y_test, y_val

def get_classes(df):
    """find claesse and the amount of classes

    Args:
        df (pd.DataFrame): all data

    Returns:
        list,int: first is list of all classes, second the length of the list
    """
    classes = df["label"].unique()
    num_classes = len(classes)
    return classes, num_classes



class SoundDataSet(Dataset):
    def __init__(self,data,label,lim=100,mode="train"):
        super().__init__()
        # self.data = data #contains file name
        self.data = data[:lim]
        self.label = label #contains class label
        self.fs = 22050 #samplerate global because all the samples have the same value
        self.n_mels = 128 #amount of mel bands fives the hight of spectrogramm
        self.n_fft = 4096 #fft length block processing
        self.hop_length = self.n_fft//2 #influnce the width of spectrogramm
        self.duration = 2 #duration tim ein seconds
        self.mode = mode #just to idetify if the samples in the data or at the beginning will be choiced
        
    def __len__(self):
        return len(self.data)
    
    #mal schauen, weil fully conected layer, kann das trainning mit 5s schnipsel und val mir2 s schnipsel gemacht werdn?
    
    #das hier auf torchaudio umbasteln
    def __getitem__(self, index):
        #calculate spectrogramm
        data, fs = librosa.load(self.data[index],mono=True,sr=self.fs)
        #resample sound to lower fs
        if fs != self.fs:
            data = librosa.resample(y=data,orig_sr=fs,target_sr=self.fs)
        #limit the length of the signal
        # data = librosa.util.fix_length(data=data,size=self.fs*self.duration) 
        if self.mode == "train":
            # data = data[self.duration*self.fs:(self.duration+1)*self.fs] #hoffnung, das vlt weiter drin im file besser unterschieden wernden kann
            # data = data[:(self.duration+3)*self.fs]
            data = librosa.util.fix_length(data=data,size=10*self.fs)#einmal ausführen lassej und schauen wie sich das auswirkt
        else:
            data = data[:self.duration*self.fs]
        #calculate stft
        S = np.abs(librosa.stft(y=data,n_fft=self.n_fft,hop_length=self.hop_length))**2
        #calculate spectrogramm
        spec = librosa.feature.melspectrogram(S=S,sr=fs,n_fft=self.n_fft,n_mels=self.n_mels,hop_length=self.hop_length,fmax=8000)#try different sizes and compare them
        #makes a spectrogramm
        db_spec = librosa.power_to_db(np.abs(spec),ref=np.max)
        #normalze spectrum
        # mean = np.mean(db_spec)
        # std = np.std(db_spec)
        # db_spec = (db_spec - mean) / std
        
        #fit spectrogramm for CNN
        spec_tensor = torch.tensor(db_spec).type(torch.float).unsqueeze(dim=0)
        #fit label for CNN
        label = torch.tensor(self.label[index])
        # print(f"shape tensor in dataset: {spec_tensor.shape}")
        return spec_tensor, label

if __name__ == "__main__":    
    #-----make a csv from the txt files
    def make_csv(path,idx):
        with open(path,"r") as f:
            data = f.readlines()
            filename = []
            label = []
            for line in range(len(data)):
                data[line] = data[line].strip().split()
                if idx == 1:
                    # path = r"C:\Users\analf\Desktop\Studium\Learn_NN\Audioscene_Classifier\Data\audio1"
                    path = os.path.join(data_path,"audio1")
                else:
                    # path = r"C:\Users\analf\Desktop\Studium\Learn_NN\Audioscene_Classifier\Data\audio"
                    path = os.path.join(data_path,"audio")
                path = f"{path}\{data[line][0]}".replace("/","\\")
                filename.append(path)
                label.append(data[line][1])
            return pd.DataFrame({"filename":filename,"label":label})

    #create dataframes to copy data inside
    all_data = pd.DataFrame()

    #transfer txt to csv
    df_audio = make_csv(info_audio_1,0)
    df_audio1 = make_csv(info_audio_2,1)
    df_segmented = pd.read_csv(info_segmented,usecols=["filename","label"])

    df_segmented["filename"] = df_segmented["filename"].str.replace("/","\\")
    df_segmented["filename"] = os.path.join(data_path,"segmented_audio") + "\\" + df_segmented["filename"].astype(str)


    def get_music_in_csv():
        df = pd.DataFrame()
        all_files = librosa.util.find_files(music_path,ext="wav")
        name = ["music" for file in all_files]
        df["filename"] = all_files
        df["label"] = name
        return df
    music_df = get_music_in_csv()

    #concat Dataframes
    all_data = pd.concat([df_audio,df_audio1,df_segmented,music_df],axis=0)
    
    #-----------encode Labels
    encoder = LabelEncoder() #das hier clt als funktion machen?
    label_encoded = encoder.fit_transform(all_data["label"])
    all_data["label"] = label_encoded
    print(encoder.classes_)
    
    
    
    #write it to a own file to have all files and labels in one file
    filename = "all_data.csv"
    all_data.sort_values(by="label").to_csv(filename,index=False)
    # inverse the encoding for further processing
    all_data["label"] = encoder.inverse_transform(all_data["label"])

    #--------- get classes
    classes, num_classes = get_classes(all_data)
    print(f"There are {len(classes)} classes and the names are: {classes}") #15 classes
    #num classes

    #amount of content per class
    class_balance = all_data["label"].value_counts()
    class_balance.plot.pie()
    #plt.show()
    #All classes are equal 625 samples


    #watch the sampling rates of all files
    def observe_fs(df):
        fs_list = []
        for x in range(len(df)):
            filename = df["filename"].iloc[x]
            fs = sf.info(filename).samplerate
            fs_list.append(fs)
        print(np.unique(np.array(fs_list))) #all have the same samplefrequenz

    observe_fs(all_data)


    def observe_length(df):
        duration_list = []
        for x in range(len(df)):
            filename = df["filename"].iloc[x]
            time = sf.info(filename).duration
            duration_list.append(time)
        duration = pd.DataFrame({"duration":duration_list})
        return duration
    #add a duration line in all data for further processing
    duration = observe_length(all_data)
    
    all_data["duration"] = duration
    print(all_data["duration"].unique())
    
    #nochmal schauen, dass alles auf 5s länge geschnitten wird und die Überflüßigen dann evtl
    #als weitere datein gespeichert werden!
    def cut_length_an_split(data:pd.DataFrame):
        #round data
        data["duration"] = data["duration"].apply(lambda x: round(x))
        #acess 30s files
        #split in 6 files a 5s with new name and safe dem 
        #acess 10s files split in 5s file 1 2files and safe dem too
        
        
        print(data.duration.unique())
 
    cut_length_an_split(all_data)
