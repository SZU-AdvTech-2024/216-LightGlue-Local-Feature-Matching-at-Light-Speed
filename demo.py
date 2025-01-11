from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet,superglue
from lightglue.utils import load_image, rbd
from lightglue import match_pair
from lightglue import viz2d
from matplotlib import pyplot as plt
# SuperPoint+LightGlue
#extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
#matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
#extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
#matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

# or ALIKED+LightGlue
#extractor = ALIKED(max_num_keypoints=2048).eval().cuda()  # load the extractor
#matcher = LightGlue(features='aliked').eval().cuda()  # load the matcher

# or SIFT+LightGlue
extractor = SIFT(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='sift').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image('/home/lys/Workplace/python/LightGlue-main/assets/DSC_0410.JPG').cuda()
image1 = load_image('/home/lys/Workplace/python/LightGlue-main/assets/DSC_0411.JPG').cuda()

# extract local features
#提取局部特征
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

# 简便的方法，直接match_pair
#feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)
# match the features

matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

#可视化特征点
# 在2D可视化工具中绘制图像

matches = matches01['matches']  # indices with shape (K,2)
points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

#可视化匹配
# 在2D可视化工具中绘制图像
axes = viz2d.plot_images([image0, image1])
# 在2D可视化工具中绘制点匹配
viz2d.plot_matches(points0, points1, color="lime", lw=0.2)
# 向2D可视化工具中添加文本信息，显示匹配层数的停止信息
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers \nDISK+LightGlue')

# 显示结果
plt.show()













