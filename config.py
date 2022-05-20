import torch
import os
#switch from "Unet-down-up-p" "Unet-down-p" "Unet-gated"
ModelType="Unet-down-up-G"
#switch from "NoGubffer", "Gubffer-Input" "Gubffer-Att" "Gbuffer-Input-Att"
# TrainingType="Gubffer-Input"
TrainingType="Gubffer-Input-Mod" 
mSkip=True
# mLossHoleAugment=True
mLossHoleArgument=1
mLossHardArgument=1

# pathPrefix = "/mnt/Disk4T/fuxihao/Medieval_WithoutDemodulate/Train/"
basePaths = [r"E:\NOAAA_Compressed\Medieval_0\\",r"E:\NOAAA_Compressed\Medieval_3\\",r"E:\NOAAA_Compressed\Medieval_4\\",r"E:\NOAAA_Compressed\Medieval_8\\"]

# Dir = "compressed.{}.npy"

# GbufferDir="gbuffer"

# AA_OccDir = "aaWarp/occ"
# No_AA_OccDir = "noaaWarp/occ"

# TAADiffuseDir="aaWarp/wrap_res"
# No_TAADiffuseDir="noaaWarp/wrap_res"

# LabelDiffuseDir="aaWarp/GT"
# NO_AA_LabelDiffuseDir="noaaWarp/GT"

# DiffuseWithoutCheckAADir="aaWarp/wrap_no_hole"
# DiffuseWithoutCheckDir="noaaWarp/wrap_no_hole"


#basePath = r"E:\NOAAA_Compressed\Medieval_0\\"
basePath = r"./"
# ValidateGbufferDir=basePath+"gbuffer"

# # No_AA_OccDir = basePath+"MedievalOrigin/Train/noaaWarp/occ"
# ValidateAA_OccDir = basePath+"aaWarp/occ"

# ValidateTAADiffuseDir=basePath+"aaWarp/wrap_res"
# # No_TAADiffuseDir=basePath+"aaWarp/noaaWarp/wrap_res_gapFrame"

# ValidateLabelDiffuseDir=basePath+"aaWarp/GT"
# # NO_AA_LabelDiffuseDir=basePath+"aaWarp/noaaWarp/GT"

# # DiffuseWithoutCheckDir=basePath+"aaWarp/noaaWarp/wrap_no_hole_gapFrame"
# ValidateDiffuseWithoutCheckAADir=basePath+"aaWarp/wrap_no_hole"

'''
TestGbufferDir=basePath+"MedievalTest/Test/gbuffer"

Test_No_AA_OccDir = basePath+"MedievalTest/Test/noaaWarp/occ"
Test_AA_OccDir = basePath+"MedievalTest/Test/aaWarp/occ"

TestTAADiffuseDir=basePath+"MedievalTest/Test/aaWarp/wrap_res_gapFrame"
TestNo_TAADiffuseDir=basePath+"MedievalTest/Test/noaaWarp/wrap_res_gapFrame"

TestLabelDiffuseDir=basePath+"MedievalTest/Test/aaWarp/GT"
Test_NO_AA_LabelDiffuseDir=basePath+"MedievalTest/Test/noaaWarp/GT"

TestDiffuseWithoutCheckDir=basePath+"MedievalTest/Test/noaaWarp/wrap_no_hole_gapFrame"
TestDiffuseWithoutCheckAADir=basePath+"MedievalTest/Test/aaWarp/wrap_no_hole_gapFrame"


TestDiffuseOFNOAADir = basePath+"MedievalTest/Test/noaaWarp/opWarp"
TestDiffuseOFAADir = basePath+"MedievalTest/Test/aaWarp/opWarp"
'''

TestDiffuseResultDir="./finalDiffuse"


NormalPrefix=r"WorldNormal"
DepthPrefix=r"SceneDepth"
RoughnessPrefix=r"Roughness"


# TestSpecularBasePrefix=r"DemoSceneSpecularColor"
# TestDiffuseBasePrefix=r"DemoSceneDiffuseColor"

warpPrefix = r"Wrap"
# noCheckHoleSpecularPrefix = r"DemoSceneWrapSpecular"





TestMetalicPrefix=r"Metallic"
TestNormalPrefix=r"WorldNormal"
TestDepthPrefix=r"SceneDepth"
TestRoughnessPrefix=r"Roughness"
mdevice=torch.device("cuda:0")

bufferChannel = 11

#Training related
learningrate=1e-3
epoch=120
printevery=200
batch_size=1



