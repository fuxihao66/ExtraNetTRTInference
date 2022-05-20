import Loaders
import Small_UnetGated
import config
import cv2 as cv
import numpy as np
from utils import ImgRead,ImgWrite, ToneSimple, DeToneSimple,ImgReadWithPrefix, ReadData
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.utils import save_image
from torch import optim
import Losses
import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter


def mLoss(holeAugment):
    if holeAugment:
        return Losses.LossHoleAugment(1)
    else:
        return F.l1_loss



def Tensor2NP(t):
    res=t.squeeze(0).cpu().numpy().transpose([1,2,0])
    res=cv.cvtColor(res,cv.COLOR_RGB2BGR)
    res = DeToneSimple(res)

    return res



def doForward(model, idx, e, indicator):
    model.eval()
    with torch.no_grad():
        img, img_2, img_3, occ_warp_img, woCheckimg, woCheckimg_2, woCheckimg_3, labelimg, metalic, Roughnessimg, Depthimg, Normalimg = ReadData(config.basePath+"compressed.{}.npy".format(idx))


        input = img
        mask = input.copy()
        mask[mask == 0.0] = 1.0
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


        mask2 = img_2.copy()
        mask2[mask2==0.]=1.0
        mask2[mask2==-1]=0.0
        mask2[mask2!=0.0]=1.0

        mask3 = img_3.copy()
        mask3[mask3==0.]=1.0
        mask3[mask3==-1]=0.0
        mask3[mask3!=0.0]=1.0

        # occ_warp_img_2[mask2!=0.0] = 0.0
        # occ_warp_img_2[occ_warp_img_2 < 0.0] = 0.0
        # occ_warp_img_2 = ToneSimple(occ_warp_img_2)
        woCheckimg_2 = ToneSimple(woCheckimg_2)

        # occ_warp_img_3[mask3!=0.0] = 0.0
        # occ_warp_img_3[occ_warp_img_3 < 0.0] = 0.0
        # occ_warp_img_3 = ToneSimple(occ_warp_img_3)
        woCheckimg_3 = ToneSimple(woCheckimg_3)

        his_1 = np.concatenate([woCheckimg, mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        his_2 = np.concatenate([woCheckimg_2, mask2[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        his_3 = np.concatenate([woCheckimg_3, mask3[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])

        hisBuffer = np.concatenate([his_3, his_2, his_1], axis=0) 



        # features = np.concatenate([Normalimg,Depthimg,Roughnessimg], axis=2)
        features = np.concatenate([Normalimg,Depthimg,Roughnessimg, metalic], axis=2)



        input = np.concatenate([woCheckimg,occ_warp_img],axis=2).transpose([2,0,1])
        labelimg = labelimg.transpose([2,0,1])
        features = features.transpose([2,0,1])
        finalMask = np.repeat(mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1)), 6, axis=2).transpose([2,0,1])

        hisBuffer = torch.tensor(hisBuffer).to(config.mdevice)
        input = torch.tensor(input).unsqueeze(0).to(config.mdevice)
        features = torch.tensor(features).unsqueeze(0).to(config.mdevice)
        finalMask=torch.tensor(finalMask).unsqueeze(0).to(config.mdevice)
        labelimg=torch.tensor(labelimg).unsqueeze(0).to(config.mdevice)


        res=model(input, features, finalMask, hisBuffer)


        cv.imwrite("./inputTrain/input train %d epoch %d.exr"%(int(idx), e), Tensor2NP(input[:,:3,:,:]) )
        cv.imwrite("./resTrain/res train %d epoch %d.exr"%(int(idx), e), Tensor2NP(res) )
        cv.imwrite("./labelTrain/label train %d epoch %d.exr"%(int(idx), e), Tensor2NP(labelimg) )   
def train(dataLoaderIns, modelSavePath):
    model = Small_UnetGated.UNetLWGated(18,3)

    model=model.to(config.mdevice)
    optimizer = optim.Adam(model.parameters(), lr=config.learningrate)


    # model_CKPT = torch.load("./totalModel.20.pth.tar")
    # model.load_state_dict(model_CKPT['state_dict'])
    # optimizer.load_state_dict(model_CKPT['optimizer'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epoch,eta_min=1e-6)

    # criterion = mLoss(holeAugment=config.mLossHoleAugment)
    criterion = Losses.mLoss()

    trainingLosses = []
    validateLosses = []
    for e in range(config.epoch):
        print("lr is ",optimizer.state_dict()['param_groups'][0]['lr'])
        model.train()

        iter=0
        loss_all=0
        startTime = time.time()
        for input,features,mask,hisBuffer,label in dataLoaderIns:
            input=input.to(config.mdevice)
            hisBuffer=hisBuffer.to(config.mdevice)
            mask=mask.to(config.mdevice)
            features=features.to(config.mdevice)
            label=label.to(config.mdevice)

            optimizer.zero_grad()


            res=model(input,features, mask, hisBuffer)
            if config.mLossHoleArgument:
                loss=criterion(res,mask,label)
            else:
                loss=criterion(res,label)
            loss.backward()
            optimizer.step()
            iter+=1
            loss_all+=loss
            if iter%config.printevery==1:
                print(loss)
        
        endTime = time.time()
        print("epoch time is {}".format(endTime - startTime))

        # if e % 10 == 0:
        #     torch.save({'epoch': e + 1, 'state_dict': model.state_dict(),
        #                     'optimizer': optimizer.state_dict()},
        #                     'model-{}.pth.tar'.format(e))
        
        print("epoch %d mean loss for train is %f"%(e,loss_all/iter))

        # lossOnValidate = inferenceOnTraining(model, 0, e)
        # print("epoch %d mean validate loss is %f"%(e,lossOnValidate))

        

        # writer.add_scalar('Loss/train', loss_all/iter, e)
        # writer.add_scalar('Loss/test', lossOnValidate, e)
        # trainingLosses.append(loss_all / iter)
        # validateLosses.append(lossOnValidate)



        if e % 5 == 0:
            doForward(model, "0181", e, 0)




        #if e % 20 == 0 and e > 50:
            #scheduler.step()
        scheduler.step()

        if e % 10 == 0:
            torch.save({'epoch': e + 1, 'state_dict': model.state_dict(), 
                        'optimizer': optimizer.state_dict()},
                         './totalModel.{}.pth.tar'.format(e))

    torch.save(model.state_dict(), modelSavePath)

    
    
    

if __name__=="__main__":
    model = Small_UnetGated.UNetLWGated_FULL(18,3)
    model.load_state_dict(torch.load("totalModel.140.pth.tar")["state_dict"])
    model=model.to(config.mdevice)
    doForward(model, "0145", 0, 0)

'''
if __name__ =="__main__":
    
    trainDiffuseDataset = Loaders.MedTrainDataset(0,augment=True)
    trainDiffuseLoader = data.DataLoader(trainDiffuseDataset,config.batch_size,shuffle=True,num_workers=2)
    
    train(trainDiffuseLoader, "./multiFrame-ultralight-skip-multiSet-noAA.pkl")
    # inference('gated_mask_baseline-multiFrame-lightMaskMul.pkl', 0)

    # train(trainSpecularLoader, "gate_v3_specular_params.pkl")
    # inference('gate_v3_specular_params.pkl', 1)
    # model = Small_UnetGated.UNetLWGated(18,3)
    # model.load_state_dict(torch.load("gated_mask_baseline-multiFrame-ultralight-skip-upsampleFlow-multiSet.pkl"))
    # model=model.to(config.mdevice)
    # model.eval()
    # for i in range(181, 200):
    #     doForward(model, str(i).zfill(4), 79, 0)
'''

