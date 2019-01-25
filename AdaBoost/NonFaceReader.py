import os
import config
import cv2
from random import randint
import random

class NonFaceReader:
    def __init__(self,dir):
        self.dir=dir
        self.fileList=list()
        listFiles = os.walk(dir)
        for root, dirs, files in listFiles:
            for file in files:
                self.fileList.append(os.path.join(root,file))

    def count(self):
        return len(self.fileList)
    
    def randomPatch(self,size=config.IMAGE_SIZE,count=None,start=0):
        if count == None:
            count=self.count()
        index=0
        random.seed(0)
        for file in self.fileList[start:-1]:
            if index >= count:
                raise StopIteration
            image=cv2.imread(file)
            rows,cols,_=image.shape
            #有些图片是灰度图
            result=None
            if (rows,cols) == size:
                result=image[:,:,0]
            else:
                x=randint(0,rows-config.IMAGE_SIZE[0])
                y=randint(0,cols-config.IMAGE_SIZE[1])
                result=image[x:x+config.IMAGE_SIZE[0],y:y+config.IMAGE_SIZE[1]]
                result=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
                (mean,stddev)=cv2.meanStdDev(result)
                if stddev<0.01: #含有的变化信息太少，不作为非人脸图片
                    continue
                result=(result-mean)/stddev
            yield result
            index+=1

if __name__=='__main__':
    root=r'E:\dataset\boostImages\NFACES'
    reader=NonFaceReader(root)
    genPatch=reader.randomPatch()
    for patch in genPatch:
        cv2.imshow("patch",patch)
        cv2.waitKey()
        cv2.destroyAllWindows()