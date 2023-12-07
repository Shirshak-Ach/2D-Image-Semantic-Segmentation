from Data.GetData import get_data
from Data.prepare_dataloader import get_from_loader
from Plots.plot_figures import plot_mask, plot_loss
from Losses.calculate_loss import DiceLoss
from Models.UNet_Architecture import Build_UNet
import torch
import torch.optim as optim

from tqdm import tqdm

from Metrics.Get_Metrics import calculate_metrics
from operator import add

import matplotlib.pyplot as plt




def training_phase(train_dataloader, test_dataloader):
    num_epochs = 2
    model = Build_UNet(num_classes=1).to(device)
    loss_function = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    overall_train_loss = overall_test_loss = []
    for epoch in range(1,num_epochs+1):
        epoch_train_loss = 0
        epoch_test_loss = 0   
        metrics_score = [0.0, 0.0]

        model.train()

        for batch_data in tqdm(train_dataloader, desc=f'Training Epoch {epoch}/{num_epochs}', unit='batch'):
            inputs = batch_data[0].to(device)
            labels = batch_data[1].to(device)
            
            # print(torch.max(inputs))
            # print(torch.min(inputs))

            # print(torch.max(labels))
            # print(torch.min(labels))
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # print(torch.max(outputs))
            # print(torch.min(outputs))
            
            train_loss = loss_function(outputs, labels)
            # train_loss.requires_grad = True
            train_loss.backward()
            overall_train_loss.append(train_loss) 
            
            score = calculate_metrics(outputs, labels)
            metrics_score = list(map(add, metrics_score, score))

            optimizer.step()
            epoch_train_loss += train_loss.item()
            
            epoch_train_loss = epoch_train_loss/len(train_dataloader)
            epoch_train_jaccard = metrics_score[0]/len(train_dataloader)
            epoch_train_acc = metrics_score[1]/len(train_dataloader)

        model.eval()
        metrics_score = [0.0, 0.0]
        with torch.no_grad():
            for batch_data in tqdm(test_dataloader, desc=f'Test Epoch {epoch}/{num_epochs}', unit='batch'):
                inputs = batch_data[0].to(device)
                labels = batch_data[1].to(device)
                outputs = model(inputs)
                
                test_loss = loss_function(outputs, labels)
                
                score = calculate_metrics(outputs, labels)
                metrics_score = list(map(add, metrics_score, score))

                optimizer.step()
                epoch_test_loss += test_loss.item()

                epoch_test_loss = epoch_train_loss/len(test_dataloader)
                epoch_test_jaccard = metrics_score[0]/len(test_dataloader)
                epoch_test_acc = metrics_score[1]/len(test_dataloader)
                
                overall_test_loss.append(test_loss) 


    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss.item():.4f}, '
          f'Train Jaccard: {epoch_train_jaccard.item():.4f}, '
          f'Train Accuracy: {epoch_train_acc.item():.4f}, '
          f'Test Loss: {test_loss.item():.4f}, '
          f'Test Jaccard: {epoch_test_jaccard.item():.4f}, '
          f'Test Accuracy: {epoch_test_acc.item():.4f}, ')
    
    return model, overall_train_loss, overall_test_loss



if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    dim = 256*2
    X_train,y_train, X_test, y_test = get_data(dim)

    train_dataloader,test_dataloader = get_from_loader(X_train, y_train, X_test, y_test)

    # print("Training Set")
    # plot_mask(X_train,y_train)
    # print("testing set")
    # plot_mask(X_test,y_test)

    model, overall_train_loss, overall_test_loss = training_phase(train_dataloader, test_dataloader)

    plot_loss(overall_train_loss,overall_test_loss)



    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_test[1]).to(device, dtype=torch.float32)
        y = torch.from_numpy(y_test[1]).to(device, dtype=torch.float32)
        x = x.unsqueeze(0)
        print(x.shape)
        x= torch.transpose(x,1,3)
        x= torch.transpose(x,2,3)
        print(x.shape)
        y_pred = model(x)
        print(y_pred.squeeze(0,1).shape)
        x = torch.squeeze(x)
        plt.imshow(x.cpu(), cmap = 'gray')
        plt.imshow(y_pred.squeeze(0,1).cpu() > 0.5, cmap = 'gray')








