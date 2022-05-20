import torch.utils.data as data
import torch
import os
import cv2 as cv
import numpy as np
import config
import time
from utils import ImgRead,ImgWrite, ToneSimple,ImgReadWithPrefix,ReadData

class MedTrainDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, typeIndicator, augment=False):  # root表示图片路径
        self.indicator = typeIndicator
        
        self.totalNum = 0
        self.augment=augment
        # self.totalSetNum = len(config.basePaths)
        
        self.imgSet = []
        for path in config.basePaths:
            imgs = os.listdir(path)
            setNum = len(imgs)
            self.imgSet.append(imgs)
            self.totalNum += setNum

    def mapIndex2PathAndIndex(self, index):
        remain = index
        for setIndex,ims in enumerate(self.imgSet):
            if remain < len(ims):
                return config.basePaths[setIndex], ims[remain].split(".")[1]
            else:
                remain -= len(ims)

        return None, -1
    def __getitem__(self, index):
        
        # idx=self.start + index
        path, idx = self.mapIndex2PathAndIndex(index)
        try:
            img, img_2, img_3, occ_warp_img, woCheckimg, woCheckimg_2, woCheckimg_3, labelimg, metalic, Roughnessimg, Depthimg, Normalimg = ReadData(path+"compressed.{}.npy".format(idx),self.augment)
        except:
            print(path)
            print(idx)

        input = img
        mask = input.copy()
        mask[mask==0.]=1.0
        mask[mask==-1]=0.0
        mask[mask!=0.0]=1.0

        # occ_warp_img[mask!=0.0] = 0.0
        occ_warp_img[occ_warp_img < 0.0] = 0.0
        woCheckimg[woCheckimg < 0.0] = 0.0
        woCheckimg_2[woCheckimg_2 < 0.0] = 0.0
        woCheckimg_3[woCheckimg_3 < 0.0] = 0.0



        labelimg = ToneSimple(labelimg)
        occ_warp_img = ToneSimple(occ_warp_img)
        woCheckimg = ToneSimple(woCheckimg)


        features = np.concatenate([Normalimg,Depthimg,Roughnessimg,metalic], axis=2)

        
        mask2 = img_2.copy()
        mask2[mask2==0.]=1.0
        mask2[mask2==-1]=0.0
        mask2[mask2!=0.0]=1.0

        mask3 = img_3.copy()
        mask3[mask3==0.]=1.0
        mask3[mask3==-1]=0.0
        mask3[mask3!=0.0]=1.0

        woCheckimg_2 = ToneSimple(woCheckimg_2)

        woCheckimg_3 = ToneSimple(woCheckimg_3)

        finalMask = np.repeat(mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1)), 6, axis=2)

        his_1 = np.concatenate([woCheckimg, mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        his_2 = np.concatenate([woCheckimg_2, mask2[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        his_3 = np.concatenate([woCheckimg_3, mask3[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])

        # endTime = time.time()
        # print("read time is {}".format(endTime - startTime))


        if config.TrainingType == "NoGubffer":
            #input,mask,label
            return torch.tensor(input.transpose([2,0,1])),torch.tensor(mask.transpose([2,0,1])),torch.tensor(labelimg.transpose([2,0,1]))
        elif config.TrainingType == "Gubffer-Input":
            #input,mask,label
            input = np.concatenate([woCheck,features],axis=2)
            return torch.tensor(input.transpose([2,0,1])),torch.tensor(finalMask.transpose([2,0,1])),torch.tensor(labelimg.transpose([2,0,1]))
        elif config.TrainingType == "Gubffer-Input-Mod":
            #input,mask,label
            input = np.concatenate([woCheckimg,occ_warp_img],axis=2)
            

            hisBuffer = np.concatenate([his_3, his_2, his_1], axis=0)

            return torch.tensor(input.transpose([2,0,1])),torch.tensor(features.transpose([2,0,1])), torch.tensor(finalMask.transpose([2,0,1])), torch.tensor(hisBuffer), torch.tensor(labelimg.transpose([2,0,1]))
        elif config.TrainingType=="Gubffer-Att":
            #input,mask,attinput,label
            return torch.tensor(input.transpose([2, 0, 1])), torch.tensor(mask.transpose([2, 0, 1])),\
                   torch.tensor(auxiBuffer.transpose([2,0,1])),torch.tensor(auxiMask.transpose([2,0,1])), torch.tensor(labelimg.transpose([2, 0, 1]))
        elif config.TrainingType == "Reweight-Gubffer-Atten":
            finalMask = mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))
            # finalMask = np.repeat(mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1)), 28, axis=2)
            inputNotGood = woCheck
            return torch.tensor(input.transpose([2, 0, 1])), torch.tensor(inputNotGood.transpose([2, 0, 1])), torch.tensor(features.transpose([2,0,1])), torch.tensor(finalMask.transpose([2, 0, 1])),\
                    torch.tensor(labelimg.transpose([2, 0, 1]))
        elif config.TrainingType=="Gbuffer-Input-Att":
            #input,mask,attinput,label
            input = np.concatenate([input,Normalimg,Depthimg,Roughnessimg],axis=2)
            attinput = np.concatenate([Normalimg,Depthimg,Roughnessimg],axis=2)
            return torch.tensor(input.transpose([2, 0, 1])), torch.tensor(mask.transpose([2, 0, 1])), \
                   torch.tensor(attinput.transpose([2, 0, 1])), torch.tensor(labelimg.transpose([2, 0, 1]))
        else:
            assert 0
    def __len__(self):
        return self.totalNum



class MedTestDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, transform=None):  # root表示图片路径
        self.TAAimgs=os.listdir(config.TestTAADir)
        self.NOAAimgs=os.listdir(config.TestNo_TAADir)

    def __getitem__(self, index):
        TAAimgstr = self.TAAimgs[index]
        NOAAimgstr = self.NOAAimgs[index]
        TAAidx = TAAimgstr.split(".")[1]
        NOAAidx = NOAAimgstr.split(".")[1]
        assert TAAidx == NOAAidx
        idx=TAAidx
        TAAimg = ImgRead(config.TestTAADir,int(idx),cvtrgb=True)
        NOTAAimg = ImgRead(config.TestNo_TAADir,int(idx),cvtrgb=True)
        if config.TrainingType =="NoGubffer":
            #TAA and NOAA image are all, pass
            pass
        elif config.TrainingType=="Gubffer-Input" or config.TrainingType=="Gubffer-Att" or config.TrainingType=="Gbuffer-Input-Att":
            Normalimg = ImgRead(config.TestGbufferDir,idx,prefix=config.TestNormalPrefix,cvtrgb=True)
            Depthimg = ImgRead(config.TestGbufferDir,idx,prefix=config.TestDepthPrefix,cvtrgb=True)
            Depthimg = (Depthimg-Depthimg.min())/(Depthimg.max()-Depthimg.min()+1e-6)
            Roughnessimg = ImgRead(config.TestGbufferDir,idx,prefix=config.TestRoughnessPrefix,cvtrgb=True)
        labelimg = ImgRead(config.TestLabelDir,idx,cvtrgb=True)

        input = np.concatenate([NOTAAimg,TAAimg],axis=2)
        #prepare mask
        mask = input.copy()
        mask[mask==-1]=0.0
        mask[mask!=0.0]=1.0
        if config.TrainingType == "NoGubffer":
            #input,mask,label
            return torch.tensor(input.transpose([2,0,1])),torch.tensor(mask.transpose([2,0,1])),torch.tensor(labelimg.transpose([2,0,1]))
        elif config.TrainingType == "Gubffer-Input":
            #input,mask,label
            input = np.concatenate([input,Normalimg,Depthimg,Roughnessimg],axis=2)
            return torch.tensor(input.transpose([2,0,1])),torch.tensor(mask.transpose([2,0,1])),torch.tensor(labelimg.transpose([2,0,1]))
        elif config.TrainingType=="Gubffer-Att":
            attinput = np.concatenate([Normalimg,Depthimg,Roughnessimg],axis=2)
            #input,mask,attinput,label
            return torch.tensor(input.transpose([2, 0, 1])), torch.tensor(mask.transpose([2, 0, 1])),\
                   torch.tensor(attinput.transpose([2,0,1])),torch.tensor(labelimg.transpose([2, 0, 1]))
        elif config.TrainingType=="Gbuffer-Input-Att":
            #input,mask,attinput,label
            input = np.concatenate([input,Normalimg,Depthimg,Roughnessimg],axis=2)
            attinput = np.concatenate([Normalimg,Depthimg,Roughnessimg],axis=2)
            return torch.tensor(input.transpose([2, 0, 1])), torch.tensor(mask.transpose([2, 0, 1])), \
                   torch.tensor(attinput.transpose([2, 0, 1])), torch.tensor(labelimg.transpose([2, 0, 1]))
        else:
            assert 0
    def __len__(self):
        return len(self.TAAimgs)
