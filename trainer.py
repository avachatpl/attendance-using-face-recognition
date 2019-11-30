from modelArch import DenseArchs
import cv2 
import numpy as np
import os,keras
from embedding import emb
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, Dense, TimeDistributed, MaxPooling1D, Flatten
import xlsxwriter
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

people=os.listdir('people')

count=0
for x in people:
    #print(x[0:len(x)-1])
    count +=1
n_classes=count
#print(n_classes)
#n_classes=3

sno=int(input('no of subjects'))
subname=[]
print('enter names')
for i in range(sno):
    sn=input()
    subname.append(sn)


e=emb()
arc=DenseArchs(n_classes)
face_model=arc.arch()
name=xlsxwriter.Workbook('attendance.xlsx')

x_data=[]
y_data=[]

for y in subname:
    worksheet=name.add_worksheet(y)
    row=0
    col=0 
    worksheet.write(col,row, "Name")
    for x in people:
        #print(x)
        for i in os.listdir('people/'+x):
            img=cv2.imread('people'+'/'+x+'/'+i,1)
            img=cv2.resize(img,(160,160))
            img=img.astype('float')/255.0
            img=np.expand_dims(img,axis=0)
            embs=e.calculate(img)
            x_data.append(embs)
            y_data.append(int(x[len(x)-2:len(x)]))

        col += 1
        print(y_data)
        worksheet.write(col,row, x[0:len(x)-2])
        
name.close()     

x_data=np.array(x_data,dtype='float')
y_data=np.array(y_data)
y_data=y_data.reshape(len(y_data),1)
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=77)


nb_filter = 4
filter_length = 5

window = x_train.shape[1]
face_model = Sequential()

face_model.add(Conv1D(filters=nb_filter,kernel_size=filter_length,activation="relu", input_shape=(window,1)))
face_model.add(MaxPooling1D())
face_model.add(Conv1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'))
face_model.add(MaxPooling1D())
face_model.add(Flatten())
face_model.add(Dense(n_classes, activation='softmax'))
face_model.summary()
face_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

face_model.fit(x_train, y_train, epochs=25, batch_size=2, validation_data=(x_test, y_test))

face_model.save('face_reco2.MODEL')
print(x_data.shape,y_data.shape)
acc=face_model.evaluate(x_test,y_test)
print('Accuracy =',acc[1]*100)

'''y=face_model.predict(x_test)
print(y)
y=np.argmax(y,axis=1)
results = confusion_matrix(x_test, y) 
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(x_test, y)) '''