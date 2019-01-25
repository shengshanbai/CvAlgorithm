from HaarFeature import *
from config import *
import numpy
import math

#只有一个特征的弱分类器
class WeekClassifer:
    def __init__(self,feature):
        self.feature=feature
        self.threshold=None
        self.output=None
        self.beta=None
        self.alpha=None

    @staticmethod
    def buildClassifers(windowSize):
        """根据window的大小，构建弱分类器列表
        """
        result=list()
        aFeatures=HaarFeatureA.buildFeatures(windowSize)
        result.extend([WeekClassifer(feature) for feature in aFeatures])
        bFeatures=HaarFeatureB.buildFeatures(windowSize)
        result.extend([WeekClassifer(feature) for feature in bFeatures])
        cFeatures=HaarFeatureC.buildFeatures(windowSize)
        result.extend([WeekClassifer(feature) for feature in cFeatures])
        dFeatures=HaarFeatureD.buildFeatures(windowSize)
        result.extend([WeekClassifer(feature) for feature in dFeatures])
        eFeatures=HaarFeatureE.buildFeatures(windowSize)
        result.extend([WeekClassifer(feature) for feature in eFeatures])
        return result

    def featureValue(self,integral):
        fvalue=self.feature.compute(integral)
        return fvalue

    def train(self,integralMap,weight,tag):
        featureArray=numpy.empty(tag.shape)
        for i in range(integralMap.shape[0]):
            featureArray[i]=self.featureValue(integralMap[i,:,:])
        idx = tag
        posWeight=weight*idx
        sumPos=posWeight.dot(featureArray)
        sumPosW=posWeight.sum()

        idx = (tag - LABEL_POSITIVE) / (-LABEL_POSITIVE)
        negWeight = weight  * idx
        sumNeg = negWeight.dot(featureArray)
        sumNegW= negWeight.sum()

        miuPos = sumPos / sumPosW
        miuNeg = sumNeg / sumNegW
        self.threshold = (miuPos + miuNeg)/2
        #选择方向
        minErrRate    = numpy.inf
        for direction in [-1, 1]:
            output=numpy.empty(featureArray.shape)
            output[featureArray * direction < self.threshold * direction]\
                    = LABEL_POSITIVE
            output[featureArray * direction >= self.threshold * direction]\
                    = LABEL_NEGATIVE
            errorRate = weight[output != tag].sum()
            if errorRate < minErrRate:
                self.bestDirection=direction
                self.output=output
                minErrRate=errorRate
        self.beta=minErrRate/(1-minErrRate)
        self.alpha=math.log(1/self.beta)
        return minErrRate
    
    def reset(self):
        self.threshold=None
        self.output=None
        self.beta=None
        self.alpha=None
    
    def predict(self,integralMap):
        output=numpy.zeros((integralMap.shape[0]))
        for index in range(integralMap.shape[0]):
            feature=self.featureValue(integralMap[index,:,:])
            if feature * self.bestDirection < self.threshold * self.bestDirection:
                output[index] = LABEL_POSITIVE
            else:
                output[index] = LABEL_NEGATIVE
        return output*self.alpha
    
    def jsonDesc(self):
        jsonDict=dict()
        jsonDict["threshold"]=self.threshold
        jsonDict["bestDirection"]=self.bestDirection
        jsonDict["alpha"]=self.alpha
        jsonDict["feature_type"]=self.feature.type
        jsonDict["feature_start"]=self.feature.pointStart
        jsonDict["feature_end"]=self.feature.pointEnd
    