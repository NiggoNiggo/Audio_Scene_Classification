import pandas as pd
from torch import nn, optim
from pre_processing import split_data, SoundDataSet,get_classes
import torch.cuda as cuda
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy,ConfusionMatrix
from CNN import CNN, train_step, test_step, eval_model
import time, torch
from tqdm import tqdm

if __name__ == '__main__':
    start_time = time.time()
    #----- device agnostic code
    device = "cuda" if cuda.is_available() else "cpu"
    

    #----------manage data
    all_data = pd.read_csv("all_data.csv",index_col=False)
    
    
    #-----encode labels
    
    #splitt val train data
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(all_data)
    
    #--------- classes 
    classes, num_classes = get_classes(all_data)

    
    #-------Model variables
    hidden_layer_1 = 32
    hidden_layer_2 = 48 #64
    batch_size = 32 #batch size änderung test #8
    num_workers = 12 #change this to your systems availible kernels
    
    #-----------Dataloader
    lim = None #none bedeutet alle elemente /set to a number for faster computation
    train_set = SoundDataSet(X_train,y_train,lim=lim)
    val_set = SoundDataSet(X_val,y_val,lim=lim)
    test_set = SoundDataSet(X_test,y_test,lim=lim)
    train_loader = DataLoader(train_set,batch_size,num_workers=num_workers)    
    test_loader = DataLoader(test_set,num_workers=num_workers)
    val_loader = DataLoader(val_set,batch_size,num_workers=num_workers)
    
    
    
    #------train variables
    epochs = 6
    lr = 0.01 #0.0001
    #early stopping oder lr regulazion machne
    
    #------metrics
    accuracy = Accuracy(task="multiclass",num_classes=num_classes).to(device)
    con_mat = ConfusionMatrix(task="multiclass",num_classes=num_classes).to(device)
    
        
    #----------init Network
    network = CNN(hidden_layer_1=hidden_layer_1,hidden_layer_2=hidden_layer_2,num_classes=num_classes).to(device)
    #load a pretrained model
    network.load_state_dict(torch.load(r"Models\model_epoch_5_acc0.82_batchsize_32_lr_0.01_fine_tuning.pkl"))
    #loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(),lr)#mal den Adam etwas genauer und wissenschaftlier anscchauen
    # optimizer = getattr(torch.optim,optimizer_choice)(network.parameters(),lr=lr)
    
    #-------Loop
    for epoch in tqdm(range(epochs)):
        train_acc, train_loss = train_step(network,train_loader,accuracy,loss_fn,optimizer,device)
        test_acc, test_loss = test_step(network,test_loader,accuracy,loss_fn,device)
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}, train_acc: {train_acc:.2f}, test_acc: {test_acc:.2f}, train_loss: {train_loss:.5f}, test_loss: {test_loss:.5f}")
        if epoch % 5 == 0:
            filename = f"Models/model_epoch_{epoch}_acc{test_acc:.2f}_batchsize_{batch_size}_lr_{lr}_fine_tuning.pkl"
            filename = f"Models/model_latest_test_acc_{test_acc:.2f}_train_acc_{train_acc:.2f}.pkl"
            torch.save(network.state_dict(),filename)
    
    #eval model
    eval_metrics = eval_model(network,val_loader,loss_fn,accuracy,con_mat,device)
    print(eval_metrics)
    
    end_time = time.time()
    last_time = end_time - start_time
    print(f"The program took: {last_time/60} min")
    
    
    ###
 