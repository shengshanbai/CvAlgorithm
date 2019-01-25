import cv2
import abc
from FaceReader import FaceReader
from NonFaceReader import NonFaceReader
from HaarFeature import *
import numpy as np
from WeekClassifer import WeekClassifer
from config import *
from StrongClassifer import StrongClassifer
import json
import os

class AdaBoost:
    def __init__(self,faceReader,nonFaceReader):
        self.imageSize=IMAGE_SIZE
        self.faceReader=faceReader
        self.nonFaceReader=nonFaceReader

    def train(self,stageCount=38):
        """
        :param stageCount:强分类器的个数
        """
        self.cascadeClassifers=list()
        for i in range(stageCount):
            strongClassifer=StrongClassifer(self.faceReader,self.nonFaceReader)
            if i <stageCount-10:
                strongClassifer.setTarget(1,0.5)
            else:
                strongClassifer.setTarget(0.99,0.3)
            #训练这个强分类器
            strongClassifer.train()
            self.saveCascade(CASCADE_FILE)
            os._exit(1)
        return

    def saveCascade(self,path):
        jsonDict=dict()
        jsonDict["cascade"] = list()
        for classifer in self.cascadeClassifers:
            jsonDict["cascade"].append(classifer.jsonDesc())
        with open(path,'w') as f:
            json.dump(jsonDict,f)

if __name__=='__main__':
    faceReader=FaceReader(r'E:\dataset\boostImages\FACES')
    nonFaceReader=NonFaceReader(r'E:\dataset\boostImages\NFACES')
    boost=AdaBoost(faceReader,nonFaceReader)
    boost.train()