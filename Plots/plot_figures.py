import numpy as np 
import matplotlib.pyplot as plt




def plot_mask(X,y):
    sample = []
    
    for i in range(6):
        left = X[i]
        right = y[i]
        combined = np.hstack((left,right))
        sample.append(combined)
        
        
    for i in range(0,6,3):

        plt.figure(figsize=(25,10))
        
        plt.subplot(2,3,1+i)
        plt.imshow(sample[i])
        
        plt.subplot(2,3,2+i)
        plt.imshow(sample[i+1])
        
        
        plt.subplot(2,3,3+i)
        plt.imshow(sample[i+2])
        
        plt.show()


def plot_loss(overall_train_loss, overall_test_loss):
    overall_train_loss = [train_loss.detach().numpy() for train_loss in overall_train_loss]
    overall_test_loss = [test_loss.detach().numpy() for test_loss in overall_test_loss]
    
    plt.plot(overall_train_loss)
    plt.plot(overall_test_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
