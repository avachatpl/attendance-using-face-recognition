from keras.models import load_model

class emb:
    def __init__(self):
        self.model=load_model('facenet_keras.h5')
        #print(self.model.inputs)
        #print(self.model.outputs)
    def calculate(self,img):
        return self.model.predict(img)[0]
