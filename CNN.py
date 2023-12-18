import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
#build model


class CNN(nn.Module):
    def __init__(self,hidden_layer_1,hidden_layer_2,num_classes):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=hidden_layer_1,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(hidden_layer_1),
            # nn.Dropout(0.4), #vlt eher batchnorm
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layer_1,out_channels=hidden_layer_1,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(hidden_layer_1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3,stride=3,padding=1) #adaptive poolinf einbauen
            nn.AdaptiveAvgPool2d(output_size=(44,73))
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layer_1,out_channels=hidden_layer_2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(hidden_layer_2),
            nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Conv2d(in_channels=hidden_layer_2,out_channels=hidden_layer_2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(hidden_layer_2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2,stride=1,padding=1)
            nn.AdaptiveAvgPool2d(output_size=(8,8)) #size ändern
            #->jetzt mit optuna nochmal ersuchen!
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_layer_2*8*8,out_features=hidden_layer_2),#44,73
            nn.BatchNorm1d(hidden_layer_2),
            nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(in_features=hidden_layer_2,out_features=num_classes)
            #hier vlt auch noch paar regulations?
        )
        
               
    def forward(self,x):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
    
#----------Train test loop

def train_step(model:nn.Module,data_loader:torch.utils.data.DataLoader,
               accuracy_fn,loss_fn:nn.Module,optimizer:optim.Optimizer,
               device:torch.cuda.device):
    model.train()
    pred_labels = []
    true_labels = []
    acc, loss = 0,0
    for idx, (data,label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device)
        y_pred = model(data)
        predictions = torch.softmax(y_pred,dim=1).argmax(dim=1)
        train_loss = loss_fn(y_pred,label) 
        train_acc = accuracy_fn(predictions,label) #hier auch nach der reihnfolge schauen 
        pred_labels.append(y_pred)
        true_labels.append(label)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        acc += train_acc
        loss += train_loss
    acc /= len(data_loader)
    loss /= len(data_loader)
    # print(f"Train acc: {acc:.2f} | Train loss: {loss:.5f}")
    return acc, loss

def test_step(model:nn.Module,data_loader:torch.utils.data.DataLoader,
               accuracy_fn,loss_fn:nn.Module,device:torch.cuda.device):
    model.eval()
    acc, loss = 0,0
    for idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device)
        with torch.inference_mode():
            y_pred = model(data)
            predictions = torch.softmax(y_pred,dim=1).argmax(dim=1)
            y_loss = loss_fn(y_pred,label)
            test_acc = accuracy_fn(predictions,label)
            loss += y_loss
            acc += test_acc
    acc = acc / len(data_loader)
    loss = loss / len(data_loader)
    # print(f"Test Acc: {acc:.2f} | Test Loss: {loss:.5f}")
    return acc, loss



#--------- Validation

def eval_model(model:torch.nn.Module,
            data_loader:torch.utils.data.DataLoader,
            loss_fn:torch.nn.Module,
            accuracy_fn,
            con_mat,
            device:torch.device):

    loss, acc = 0,0
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device), y.type(torch.LongTensor).to(device)
            y_pred = model(X)
            #hinzufügen der predictuons and labels fpr the confiusmatrix
            true_labels.extend(y)
            predicted_labels.extend(torch.softmax(y_pred,dim=1).argmax(dim=1))
            
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(torch.softmax(y_pred,dim=1).argmax(dim=1),y)
            
            
        loss /= len(data_loader)
        acc /= len(data_loader)
    con_mat(torch.Tensor(predicted_labels).to(device),torch.Tensor(true_labels).to(device))
    con_mat.plot()
    plt.show()
    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc":acc.item()}

