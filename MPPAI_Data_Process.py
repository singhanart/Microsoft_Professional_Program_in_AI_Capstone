import PIL
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import imshow

def imageprepare(argv):
    i = PIL.Image.open(argv).convert('L')
    tv = list(i.getdata())
    return tv

def imagetoPixel (idnumber,amount,directory,idrow,valarray):
    stop = int(idnumber + amount)
    while idnumber < stop:
        idname = 'id_' + str(idnumber)
        idrow.append(idname)
        file = os.path.join(directory, str(idnumber) + str('.png'))
        try:
            x = imageprepare(file)
            npX = np.array(x)
            norX = (npX / 225).round(4)
            valarray.append(norX)
        except FileNotFoundError:
            print('File out of index')
            del (idrow[-1])
            break
        idnumber += 1
    return idrow,valarray
    print(idrow)
    print('Len of idrow ' + str(len(idrow)))

def createDataFrame(idrow,valarray,colList,filename):
    df = pd.DataFrame(np.asarray(valarray), index = idrow, columns = colList)
    df.to_csv(filename, sep=',', index=True, header=True)

def filePathFinder(namelist,pathdict):
    for i in namelist:
        lo = os.path.join(os.getcwd(), i)
        pathdict[i] = lo

#Get file path of train & test images folders and train_label file
locationList = ['train_images', 'test_images','train_label.csv']
filePathDict ={}
filePathFinder(locationList,filePathDict)

# create column name
colList = []
for i in range(1, 4097):
    text = 'Pixel_' + str(i)
    colList.append(text)

#Process images to CSV and concat lable to training images csv
for i in filePathDict:
    if i == 'train_images':
        train_Id = []
        train_ValList_to_Array = []
        imagetoPixel(100000,20000, filePathDict.get(i),train_Id,train_ValList_to_Array)
        createDataFrame(train_Id,train_ValList_to_Array,colList,'train_pixel_data.csv')

    elif i == 'test_images':
        test_Id =[]
        test_ValList_to_Array = []
        imagetoPixel(200000,20000, filePathDict.get(i),test_Id,test_ValList_to_Array)
        createDataFrame(test_Id,test_ValList_to_Array,colList,'test_pixel_data.csv')

    elif i == 'train_label.csv':
        Px_Data_CSV = pd.read_csv('train_pixel_data.csv')
        label_CSV = pd.read_csv(filePathDict.get(i))
        added_Label = pd.concat([Px_Data_CSV,label_CSV],axis = 1)
        added_Label.to_csv("train_pixel_data_with_label.csv", sep=',', index=True, header=True)
    else:
        pass