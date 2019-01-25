from config import *
import cv2
import numpy as np
from WeekClassifer import WeekClassifer
import os
import copy

class StrongClassifer:
    def __init__(self,faceReader,nonFaceReader):
        #调节强分类器的参数，范围是0-1
        self.beta=1.0
        self.threshold=0
        self.classifers=list()
        self.faceReader=faceReader
        self.nonFaceReader=nonFaceReader

    def setTarget(self,minReCallRate,maxFalsePos):
        """设置强分类器的目标
        :param minReCallRate,最小的召回率
        :param maxFalsePos,最大错判率
        """
        self.minReCallRate=minReCallRate
        self.maxFalsePos=maxFalsePos

    def train(self):
        """训练当前的强分类器
        """
        #留下1000张人脸做验证集
        self.threshold=0.0
        trainPosCount=self.faceReader.count()
        trainNegCount=NON_FACE_COUNT
        #初始化参数训练用权重矩阵
        weight=np.array([1/(2*trainPosCount) for i in range(trainPosCount)])
        weight=np.append(weight,[1/(2*trainNegCount) for i in range(trainNegCount)])
        #初始化标签矩阵
        tag=np.array([LABEL_POSITIVE for i in range(trainPosCount)])
        tag=np.append(tag,[LABEL_NEGATIVE for i in range(trainNegCount)])
        #初始化标签数据
        allClassifer=WeekClassifer.buildClassifers(IMAGE_SIZE)
        #求取所有图像的积分图
        integralMap=np.empty((trainPosCount+trainNegCount,IMAGE_SIZE[0]+1,IMAGE_SIZE[1]+1))
        if os.path.exists(FACE_TRAIN_INTMAP):
            faceMap=np.load(FACE_TRAIN_INTMAP)
            integralMap[0:trainPosCount]=faceMap
        else:
            faceGen=self.faceReader.genFaceImage(count=trainPosCount)
            for index,face in enumerate(faceGen):
                    integral=cv2.integral(face)
                    integralMap[index,:,:]=integral
                    del face
                    del integral
            np.save(FACE_TRAIN_INTMAP,integralMap[0:trainPosCount,:,:])
        print("face integral map computed！")
        nonFaceGen=self.nonFaceReader.randomPatch(count=trainNegCount)
        for i,image in enumerate(nonFaceGen):
            integral=cv2.integral(image)
            integralMap[i+trainPosCount,:,:]=integral
            del image
            del integral
        print("all integral map computed！")
        #训练分类器，直到达到预设的目标
        while not self.isErrDesired(integralMap,tag,trainPosCount):
            #归一化权重
            weight=weight/np.sum(weight)
            #训练所有图片,选出最优分类器
            minErrRate = np.inf
            bestClassifer=None
            for classifer in allClassifer:
                errRate=classifer.train(integralMap,weight,tag)
                if errRate < minErrRate:
                    bestClassifer=classifer
                    minErrRate=errRate
            self.classifers.append(bestClassifer)
            self.threshold+=1/2*bestClassifer.alpha
            allClassifer.remove(bestClassifer)
            #更新权重
            correctLoc=np.where(bestClassifer.output == tag)
            for loc in correctLoc:
                weight[loc]*=bestClassifer.beta

    def isErrDesired(self,integralMap,tag,posCount):
        """测试当前的强分类器，是否以满足设定的目标
        """
        if not self.classifers:#当前没有分类器
            return False
        #对不同的beta条件，测试当前分类器是否达到目标
        for beta in np.arange(1.0,0,-0.02):
            sumFeature=np.zeros(tag.shape)
            for classifer in self.classifers:
                sumFeature+=classifer.predict(integralMap)
            result=np.array([LABEL_POSITIVE if feature>=beta*self.threshold else LABEL_NEGATIVE for feature in sumFeature])
            posRecallRate=np.size(np.where(result[0:posCount]==LABEL_POSITIVE))/posCount
            falsePosRate=np.size(np.where(result[posCount:-1]==LABEL_NEGATIVE))/NON_FACE_COUNT
            print("with beta:%f posRecallRate:%f falsePosRate:%f" %(beta,posRecallRate,falsePosRate))
            if posRecallRate < self.minReCallRate:
                continue

            if falsePosRate > self.maxFalsePos:
                return False
            self.beta=beta
            print("find target classfer with beta:%f threshold %f" % (self.beta,self.threshold))
            return True
        return False
    
    def jsonDesc(self):
        jsonDict=dict()
        jsonDict["beta"]=self.beta
        jsonDict["threshold"]=self.threshold
        jsonDict["weakClassifer"]=list()
        for classifer in self.classifers:
            jsonDict["weakClassifer"].append(classifer.jsonDesc())
        return jsonDict

    def predict(self,image):
        integral=cv2.integral(image)
        for classifer in self.classifers:
            if classifer.predict(integral) == LABEL_NEGATIVE:
                return LABEL_NEGATIVE
        return LABEL_POSITIVE
    
    def empty(self):
        return not self.classifers