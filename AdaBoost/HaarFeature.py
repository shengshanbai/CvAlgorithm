import abc
import cv2

#haar特征的表示类
FEATURE_A=1
FEATURE_B=2
FEATURE_C=3
FEATURE_D=4
FEATURE_E=4
class HaarFeature:
    def __init__(self,pointStart,pointEnd):
        self.pointStart=pointStart
        self.pointEnd=pointEnd
        self.type=None

    #计算特征值
    @abc.abstractmethod
    def compute(self,integral):
        pass

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.pointStart == other.pointStart and self.pointEnd == other.pointEnd
    
    def __hash__(self):
        return hash((type(self),self.pointStart,self.pointEnd))
    
    @staticmethod
    def buildFeatures(cls,windowSize,minW,minH):
        """根据窗口大小返回所有可能的feature
        """
        features=set()
        #对不同大小的feature遍历
        for w in range(windowSize[0]//minW):
            for h in range(windowSize[1]//minH):
                width=minW*(w+1)
                height=minH*(h+1)
                #不同的起始位置的feature遍历
                for i in range(windowSize[0]-width+1):
                    for j in range(windowSize[1]-height+1):
                        feature=cls((i,j),(i+width,j+height))
                        features.add(feature)
        return features

    def __repr__(self):
        return "type:%s,pointStart:%r pointEnd:%r" % (self.__class__,self.pointStart,self.pointEnd)

    @staticmethod
    def integralRect(integral,point1,point2):
        """计算包含在point1和point2的矩形框中的积分,包含point1，不包含point2
        """
        #因为要使用积分图，所以坐标都加1，以包含自己
        point3=(point1[0],point2[1])
        point4=(point2[0],point1[1])
        return integral[point2[0],point2[1]]+integral[point1[0],point1[1]]-integral[point3[0],point3[1]]-integral[point4[0],point4[1]]

#根据论文定义的几种haar特征
class HaarFeatureA(HaarFeature):
    def __init__(self,pointStart,pointEnd):
        super().__init__(pointStart,pointEnd)
        self.type=FEATURE_A

    def compute(self,integral):
        intWhite=HaarFeature.integralRect(integral,
            self.pointStart,
            (self.pointEnd[0],(self.pointStart[1]+self.pointEnd[1])//2))
        intBlack=HaarFeature.integralRect(integral,
            (self.pointStart[0],(self.pointStart[1]+self.pointEnd[1])//2),
            self.pointEnd)
        return intWhite-intBlack

    @classmethod
    def buildFeatures(cls,windowSize):
        """根据窗口大小返回所有可能的feature
        """
        return HaarFeature.buildFeatures(cls,windowSize,2,1)

class HaarFeatureB(HaarFeature):
    def __init__(self,pointStart,pointEnd):
        super().__init__(pointStart,pointEnd)
        self.type=FEATURE_B

    def compute(self,integral):
        intWhite=HaarFeature.integralRect(integral,
            ((self.pointStart[0]+self.pointEnd[0])//2,self.pointStart[1]),
            self.pointEnd)
        intBlack=HaarFeature.integralRect(integral,
            self.pointStart,
            ((self.pointStart[0]+self.pointEnd[0])//2,self.pointEnd[1]))
        return intWhite-intBlack

    @classmethod
    def buildFeatures(cls,windowSize):
        """根据窗口大小返回所有可能的feature
        """
        return HaarFeature.buildFeatures(cls,windowSize,1,2)

class HaarFeatureC(HaarFeature):
    def __init__(self,pointStart,pointEnd):
        super().__init__(pointStart,pointEnd)
        self.type=FEATURE_C

    def compute(self,integral):
        intWhite=HaarFeature.integralRect(integral,
            self.pointStart,
            (self.pointEnd[0],(2*self.pointStart[1]+self.pointEnd[1])//3))
        intWhite += HaarFeature.integralRect(integral,
            (self.pointStart[0],(self.pointStart[1]+2*self.pointEnd[1])//3),
            self.pointEnd)
        intBlack=HaarFeature.integralRect(integral,
            (self.pointStart[0],(2*self.pointStart[1]+self.pointEnd[1])//3),
            (self.pointEnd[0],(self.pointStart[1]+2*self.pointEnd[1])//3))
        return intWhite-intBlack
        
    @classmethod
    def buildFeatures(cls,windowSize):
        """根据窗口大小返回所有可能的feature
        """
        return HaarFeature.buildFeatures(cls,windowSize,3,1)

class HaarFeatureD(HaarFeature):
    def __init__(self,pointStart,pointEnd):
        super().__init__(pointStart,pointEnd)
        self.type=FEATURE_D

    def compute(self,integral):
        intWhite=HaarFeature.integralRect(integral,
            self.pointStart,
            [(x+y)//2 for x,y in zip(self.pointStart,self.pointEnd)])

        intWhite +=HaarFeature.integralRect(integral,
            [(x+y)//2 for x,y in zip(self.pointStart,self.pointEnd)],
            self.pointEnd)
        intBlack = HaarFeature.integralRect(integral,
            (self.pointStart[0],(self.pointStart[1]+self.pointEnd[1])//2),
            ((self.pointStart[0]+self.pointEnd[0])//2,self.pointEnd[1]))
        intBlack += HaarFeature.integralRect(integral,
            ((self.pointStart[0]+self.pointEnd[0])//2,self.pointStart[1]),
            (self.pointEnd[0],(self.pointStart[1]+self.pointEnd[1])//2))
        return intWhite-intBlack

    @classmethod
    def buildFeatures(cls,windowSize):
        """根据窗口大小返回所有可能的feature
        """
        return HaarFeature.buildFeatures(cls,windowSize,2,2)

class HaarFeatureE(HaarFeature):
    def __init__(self,pointStart,pointEnd):
        super().__init__(pointStart,pointEnd)
        self.type=FEATURE_E

    def compute(self,integral):
        intWhite=HaarFeature.integralRect(integral,
            self.pointStart,
            ((2*self.pointStart[0]+self.pointEnd[0])//3,self.pointEnd[1]))
        intWhite += HaarFeature.integralRect(integral,
            ((self.pointStart[0]+2*self.pointEnd[0])//3,self.pointStart[1]),
            self.pointEnd)
        intBlack=HaarFeature.integralRect(integral,
            ((2*self.pointStart[0]+self.pointEnd[0])//3,self.pointStart[1]),
            ((self.pointStart[0]+2*self.pointEnd[0])//3,self.pointEnd[1]))
        return intWhite-intBlack
        
    @classmethod
    def buildFeatures(cls,windowSize):
        """根据窗口大小返回所有可能的feature
        """
        return HaarFeature.buildFeatures(cls,windowSize,1,3)

if __name__=='__main__':
    features = HaarFeatureD.buildFeatures((24,24))
    print('feature a count:',len(features))