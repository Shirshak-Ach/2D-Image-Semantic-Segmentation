
import os
from tqdm import tqdm 
import cv2
import numpy as np 


def get_image_and_mask_array(X_shape,training_files, testing_files,image_path,mask_path, flag = "test"):
    im_array = []
    mask_array = []
    
    if flag == "test":
        print("Extraing Data for testing")
        for i in tqdm(testing_files[:10]): 
            im = cv2.resize(cv2.imread(os.path.join(image_path,i)),(X_shape,X_shape))[:,:,0]
            mask = cv2.resize(cv2.imread(os.path.join(mask_path,i)),(X_shape,X_shape))[:,:,0]
            
            im_array.append(im)
            mask_array.append(mask)
        
        return im_array,mask_array
    
    if flag == "train":
        print("Extraing Data for training")
        for i in tqdm(training_files[:10]): 
            im = cv2.resize(cv2.imread(os.path.join(image_path,i.split("_mask")[0]+".png")),(X_shape,X_shape))[:,:,0]
            mask = cv2.resize(cv2.imread(os.path.join(mask_path,i+".png")),(X_shape,X_shape))[:,:,0]

            im_array.append(im)
            mask_array.append(mask)

        return im_array,mask_array
    

def get_data(dim = 256*2):
    
    image_path = '/media/shirshak/E076749B767473DE/Lung Segmentation Dataset/Lung Segmentation/CXR_png'
    mask_path = '/media/shirshak/E076749B767473DE/Lung Segmentation Dataset/Lung Segmentation/masks'

    images = os.listdir(image_path)
    mask = os.listdir(mask_path)


    mask = [single_name.split(".png")[0] for single_name in mask]
    image_file_name = [single_name.split("_mask")[0] for single_name in mask]



    check = [i for i in mask if "mask" in i]
    # print("Total mask that has modified name:",len(check))

    # print(len(os.listdir(image_path)))
    # print(len(set(os.listdir(image_path)) & set(os.listdir(mask_path))))


    training_files = check
    testing_files = list(set(os.listdir(image_path)) & set(os.listdir(mask_path)))

    # print(len(testing_files[:10]))
    # print(len(training_files[:10]))
    
    X_train,y_train = get_image_and_mask_array(dim,training_files,testing_files,image_path,mask_path, flag="train")
    X_test, y_test = get_image_and_mask_array(dim,training_files,testing_files,image_path,mask_path)

    X_train = np.array(X_train).reshape(len(X_train),dim,dim,1)
    y_train = np.array(y_train).reshape(len(y_train),dim,dim,1)
    X_test = np.array(X_test).reshape(len(X_test),dim,dim,1)
    y_test = np.array(y_test).reshape(len(y_test),dim,dim,1)
    assert X_train.shape == y_train.shape
    assert X_test.shape == y_test.shape
    
    return X_train, y_train, X_test, y_test















