import os
import collections
import cv2
import config

class FaceReader:
    def __init__(self,faceDir):
        self.root=faceDir
        self.datas=list()
        listFiles=os.walk(self.root)
        for root, dirs, files in listFiles:
            for file in files:
                self.datas.append(os.path.join(self.root,file))

    def count(self):
        return len(self.datas)

    def genFaceImage(self,count,size=config.IMAGE_SIZE,start=0):
        index=0
        for item in self.datas[start:-1]:
            if index >= count:
                raise StopIteration
            face = cv2.imread(item)
            rows,cols,channel=face.shape
            if channel !=1:
                face=face[:,:,1]
            (mean,stddev)=cv2.meanStdDev(face)
            face=(face-mean)/stddev
            yield face
            index+=1
                

if __name__ == '__main__':
    reader=FaceReader(r'E:\dataset\boostImages\FACES')
    faceGen=reader.genFaceImage(2)
    while True:
        face=next(faceGen)
