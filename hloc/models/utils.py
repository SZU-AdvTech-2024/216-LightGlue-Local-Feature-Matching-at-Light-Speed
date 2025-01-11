# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, False)
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file)
        self.i = self.i + 1
        # return (image, True)
        return (image, True, image_file.rsplit('/', 1)[-1])

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()

def world2img():
    white = (255, 255, 255)
    black = (0, 0, 0)

    # 读取图像
    img=cv2.imread('/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf/59-nr_fine.jpg')

    # 内参
    fx, fy, cx, cy = 719.3115,719.3115,360,480

    # 外参
    R = [[0.999,0.0135,-0.0105],
                 [-0.01,0.97,0.24],
                 [0.0135,-0.2416,0.9702]]
    R = np.array(R)
    p3d_w=[]
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf/point.txt","r") as file:
        for line in file:
            print(line)
            data=line.split()
            if len(data) >= 3:
                x,y,z=map(float,data[:3])
                p3d_w.append([x,y,z])
    t = np.array([-0.582, -4.025, 0.288])


    gt=[]
    for p in p3d_w:
        pc=np.matmul(R,p)+t
        u=int(pc[0]/pc[2]*fx+cx)
        v=int(pc[1]/pc[2]*fy+cy)

        gt.append((u,v))
        cv2.circle(img, (u, v), 5, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (u, v), 3, white, -1, lineType=cv2.LINE_AA)
    cv2.imwrite("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf/nerf_pro.png", img)

    print("")

def img2world():
    img=cv2.imread('/home/lys/Workplace/python/SuperGluePretrainedNetwork/input/1_477.jpg')
    array=np.squeeze(img[:,:,:1])
     # nerf影像内参
    fx, fy, cx, cy = 1015.6,1015.6,960,540
    R = np.array([[0.005704150070, -0.000368164212, -0.015231980011],
                    [-0.007911253721, 0.013828545809, -0.003296889365],
                    [0.013021551073, 0.008562819101, 0.004669409245]])
    t = np.array([3.146306037903, 1.731960296631, -0.371804714203])

    points=[]
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/depth.txt","w") as f1:
        for i in range(len(array)):
            for j in range(len(array[i])):
                depth=array[i][j]

                if depth>=240:
                    continue
                x_normalized=(j-cx)/fx
                y_normalized=(i-cy)/fy

                # 转换为三维坐标
                X = x_normalized * depth
                Y = y_normalized * depth
                Z = depth

                # 相机坐标系转世界坐标系
                tem=[X,Y,Z]
                pw=np.matmul(R,tem)+t
                # tem=[X-t[0],Y-t[1],Z-t[2]]
                # pw=np.matmul(R.T,tem)
                # 保存世界坐标系下的坐标
                f1.write(str(pw[0])+" "+str(pw[1])+" "+str(pw[2])+"\n")
                # 保存相机坐标系下的坐标
                # f1.write(str(X)+" "+str(Y)+" "+str(Z)+"\n")
                # f1.write(str(X)+" "+str(Y)+" "+str(Z)+"\n")

    print("已生成深度图")
# def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
#                             mkpts1, color, text, path=None,
#                             show_keypoints=False, margin=10,
#                             opencv_display=False, opencv_title='',
#                             small_text=[]):
def make_matching_plot_fast(imgName, image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
#问题：这个传参入口在哪
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    # ------------
    # m0为已匹配的到第一张图片的像素特征坐标
    m0=np.round(mkpts0).astype(int) # 左侧特征点像素坐标
    m1=np.round(mkpts1).astype(int) # 右侧特征点像素坐标

    # 读取Colmap深度图
    # depth_path = "/home/lys/Workplace/python/SuperGluePretrainedNetwork/second/DJI_0173.JPG.photometric.bin"
    # depth_map = read_array(depth_path) 

    # 读取图像形式深度图
    depth_map = cv2.imread('/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/dbImgsDepth/img_116.jpg')
    depth_map= cv2.resize(depth_map, (640, 480)) # 如果输入参数为 resize = -1 则将深度图进行下采样
    depth_map=np.squeeze(depth_map[:,:,:1])
    # read_depth(depth_map) # 将深度图写成txt格式导入导入至cc中展示

    pts2D0=[]
    pts2D1=[]
    for a,b in zip(m0,m1):
        if depth_map[a[1]][a[0]]==0:
            continue
        pts2D0.append((a[0],a[1]))
        pts2D1.append((b[0],b[1]))
    

    # 绘制img0
    white = (255, 255, 255)
    black = (0, 0, 0)
    img=255*np.ones((H0, W0), np.uint8)
    img[:H0, :W0] = image0
    

    # NeRF相机内参及变换矩阵
    fx, fy, cx, cy = 338.533, 451.377, 320, 240
    R = np.array([[0.00451975, -0.00133227, -0.01447669],
                    [-0.0071004,  0.01302671, -0.00341564],
                    [0.01268596,  0.00776578,  0.003246]])
    t = np.array([2.87390404, 2.43341367, 0.933251644])

    # 图像坐标系->相机坐标系
    p3d=pixel_to_3d(pts2D0,depth_map,fx,fy,cx,cy)
    p3d_w=p3dcam2world(p3d,R,t)

    # 将2D点保存成txt文件和图片展示
    img2=255*np.ones((H1, W1), np.uint8)
    img2[:H1, :W1] = image1
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/imgPairs/"+imgName.rsplit('.', 1)[0]+"p2d.txt","w") as f:
        for x,y in pts2D1:
            f.write(str(x)+' '+ str(y)+'\n')
            cv2.circle(img2, (int(x), int(y)), 5, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(img2, (int(x), int(y)), 3, white, -1, lineType=cv2.LINE_AA)
    cv2.imwrite('/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/imgPairs/'+imgName,img2)

    # 将3D点保存成txt格式
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/imgPairs/"+imgName.rsplit('.', 1)[0]+"p3d.txt","w") as f1:
        for x,y,z in p3d_w:
            f1.write(f"{x} {y} {z}\n")


    # 地面图像相关参数（内外参）
    img2=cv2.imread('/home/lys/Workplace/python/SuperGluePretrainedNetwork/input/2_8686.jpg',cv2.IMREAD_GRAYSCALE)
    img2=cv2.resize(img2,(640,480))

    # 地面影像相机内参
    fx, fy, cx, cy = 465.629,524.413,320.0,240.0
    qvec= 0.7401215430968716, 0.2758617824057936, -0.609836835053153, -0.06495700935355284
    R=qvec2rotmat(qvec)
    t=np.array((3.080471076947961, 3.0873279979788646, 0.4430963499465381))


    pts2D_0to1=[]
    for p in p3d_w:
        pc=np.matmul(R.T,p-t)
        u=int(pc[0]/pc[2]*fx+cx)
        v=int(pc[1]/pc[2]*fy+cy)
        pts2D_0to1.append((u,v))

    # for x,y in pts2D_0to1:
    #     cv2.circle(img2, (int(x), int(y)), 5, black, -1, lineType=cv2.LINE_AA)
    #     cv2.circle(img2, (int(x), int(y)), 3, white, -1, lineType=cv2.LINE_AA)
    # cv2.circle(img2, (6, 8), 3, white, -1, lineType=cv2.LINE_AA)
    # cv2.imwrite('/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/IMG_8656_after.jpg',img2)
    for (x0,y0),(x1,y1) in zip(pts2D_0to1,pts2D1):
        cv2.line(img2,(x0,y0),(x1,y1),color=(0,255,0),thickness=1,lineType=cv2.LINE_AA)
        cv2.circle(img2, (x0, y0), 2, -1, lineType=cv2.LINE_AA)
        cv2.circle(img2, (x1, y1), 2, -1,lineType=cv2.LINE_AA)
    cv2.imwrite('/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/after.jpg',img2)
    pixel_errors=compute_pixel_error(pts2D_0to1, pts2D1)

   
    # 统计
    thresh = np.array([10, 20, 40, 80, 100])
    counts = range_count(thresh, pixel_errors) 
    for value, count in counts.items():
        print(f"0到{value}范围内的元素个数：", count)
    # out = 255*np.ones((H, W), np.uint8)
    # out[:H0, :W0] = img2
    # out[:H1, W0+margin:] = image1
    # out = np.stack([out]*3, -1)
    # ---------
    
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)
            
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    # mkpts0=gt
    # mkpts0=points
    # mkpts1=points1
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)


    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out

# 相机坐标系至世界坐标系
def p3dcam2world(p3dcam,R,t):
    p3d_w=[]
    for x,y,z in p3dcam:
        # nerf--
        tem=[x,y,z]
        pw=np.matmul(R,tem)+t
        p3d_w.append(pw)
    return p3d_w

# 读取深度图
def read_depth(array):
    from scipy.ndimage import median_filter

    # 空中影像内参
    # fx, fy, cx, cy = 1348.914, 1346.358, 1000, 665.693

    # nerf影像内参
    # fx, fy, cx, cy = 1015.6,1015.6,960,540  # nerf深度图原尺寸内参
    fx, fy, cx, cy = 338.533, 451.377, 320, 240
    R = np.array([[0.005704150070, -0.000368164212, -0.015231980011],
                    [-0.007911253721, 0.013828545809, -0.003296889365],
                    [0.013021551073, 0.008562819101, 0.004669409245]])
    t = np.array([3.146306037903, 1.731960296631, -0.371804714203])

    points=[]
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/depth.txt","w") as f1:
        for i in range(len(array)):
            for j in range(len(array[i])):
                depth=array[i][j]
                if depth==0 or depth>=250:
                    continue
                x_normalized=(j-cx)/fx
                y_normalized=(i-cy)/fy

                # 转换为三维坐标
                X = x_normalized * depth
                Y = y_normalized * depth
                Z = depth

                # 相机坐标系转世界坐标系
                tem=[X,Y,Z]
                pw=np.matmul(R,tem)+t
                # print(str(pw[0])+" "+str(pw[1])+" "+str(pw[2])+"\n")
                f1.write(str(pw[0])+" "+str(pw[1])+" "+str(pw[2])+"\n")
                # f1.write(str(X)+" "+str(Y)+" "+str(Z)+"\n")


# 读取Colmap的深度图
def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

# 像素坐标->相机坐标
def pixel_to_3d(points, depth_map, fx, fy, cx, cy):
    p3d=[]
    for u,v in points:
        depth=depth_map[v][u]

        # 转换为归一化平面坐标
        x_normalized=(u-cx)/fx
        y_normalized=(v-cy)/fy

        # 转换为三维坐标
        X = x_normalized * depth
        Y = y_normalized * depth
        Z = depth
        p3d.append((X,Y,Z))
    
    return p3d


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)

# 将地面colmap坐标系转换到空中倾斜模型的colmap坐标系
def groundcord2aircord():
    zz = np.array([[0, 0, 0, 1]])
    trans = np.array([[0.06206447, -0.08907884, -0.30995733,  1.74574546],
                       [ -0.13346705,  0.28025171, -0.10726657,  3.11747802],
                       [ 0.29359007,  0.14623451,  0.01676073,  0.40122953],
                        [0, 0, 0, 1]])
#问题：这个参数在哪 地面转换到空中的配准影像

    lines=[]
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/地面imgs文件.txt",'r') as f:
        lines=f.readlines()
    
    new_cord=[]
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/地面imgs文件已转换.txt",'w') as f:
        for line in lines:
            if "jpg" not in line:
                continue
            line = line.rstrip().split(' ')
            imgName = line[-1]
            qvec = [float(x) for x in line[1:5]]
            tvec = [float(x) for x in line[5:8]]
            R=qvec2rotmat(qvec)
            tvec = -R.T @ tvec
            R = R.T

            # 地面世界坐标转空中世界坐标
            tem1 = np.c_[R, tvec]
            tem2 = np.r_[tem1, zz]
            final = np.matmul(trans, tem2)
            t = final[:-1, -1]
            R = final[:-1, :-1]
            new_cord.append(t)

            # 世界坐标转相机坐标   
            qvec_new=rotmat2qvec(R)

            f.write(f"{imgName} {qvec_new[0]} {qvec_new[1]} {qvec_new[2]} {qvec_new[3]} {t[0]} {t[1]} {t[2]}\n")

    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/new_ground_cord_test.txt",'w') as f:
        for x,y,z in new_cord:
            f.write(f"{x} {y} {z}\n")
    print("")

def compute_pixel_error(points1, points2):
    """
    计算两组对应点的像素误差

    参数：
    points1：第一组点的坐标，每个点的坐标为(x, y)
    points2：第二组点的坐标，每个点的坐标为(x, y)

    返回值：
    pixel_errors：每个对应点的像素误差，为一个数组
    """
    points1 = np.array(points1)
    points2 = np.array(points2)
    pixel_errors = np.sqrt(np.sum((points1 - points2)**2, axis=1))
    return pixel_errors


def count_elements_in_range(array, start, end):
    """
    统计数组中落在给定范围内的元素个数

    参数:
    array: 要统计的数组
    start: 范围的起始值
    end: 范围的结束值

    返回值:
    count: 落在给定范围内的元素个数
    """

    count = np.count_nonzero((array >= start) & (array < end))
    return count

def range_count(target_array, count_array):
    """
    统计从0到目标数组中每个元素值的范围内，计数数组中的元素个数

    参数：
    target_array：目标数组，用于定义范围
    count_array：要统计的数组

    返回值：
    counts：一个字典，键是范围的结束值，值是范围内计数数组的元素个数
    """

    counts = {}
    for value in target_array:
        count = count_elements_in_range(count_array, 0, value + 1)
        counts[value] = count
    return counts

# 处理image和camera文件
def processImgandCam():
    from collections import defaultdict
    imgLine={}
    camLine={}
    imgMap = defaultdict(list)
    # 最终的分辨率
    width,heigth=640,480

    # 处理cam文件
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/back/cameras.txt",'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            line=line.rstrip('\n').split(' ')
            id=line[0]
            w=float(line[2])
            h=float(line[3])
            scale_x=w/width
            scale_y=h/heigth
            fx=float(line[4])/scale_x
            fy=float(line[5])/scale_y
            cx=float(line[6])/scale_x
            cy=float(line[7])/scale_y
            camLine[id]=f"{fx:.3f},{fy:.3f},{cx:.1f},{cy:.1f}"
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/back/images.txt",'r') as f:
        lines = f.readlines()
        for line in lines:
            if ".jpg" not in line:
                continue
            line=line.rstrip('\n').split(" ")
            id=line[0]
            imgName=line[-1]
            imgMap[imgName].append(camLine[id])

    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/地面imgs文件已转换.txt",'r') as f:
        lines=f.readlines()
        for line in lines:
            imgName,rt=line.rstrip('\n').split(' ',1)
            rt=rt.replace(' ',', ')
            imgMap[imgName].append(rt)
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/地面图像内参及真实位姿.txt",'w') as f:
        for item in imgMap.items():
            f.write(f"图像名称：{item[0]}\n")
            f.write(f"图像内参：{item[1][0]}\n")
            f.write(f"图像位姿：{item[1][1]}\n\n")
    print("")

def returnimgMap():
    from collections import defaultdict
    imgLine={}
    camLine={}
    imgMap = defaultdict(list)
    # 最终的分辨率
    width,heigth=640,480

    # 处理cam文件
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/back/cameras.txt",'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            line=line.rstrip('\n').split(' ')
            id=line[0]
            w=float(line[2])
            h=float(line[3])
            scale_x=w/width
            scale_y=h/heigth
            fx=float(line[4])/scale_x
            fy=float(line[5])/scale_y
            cx=float(line[6])/scale_x
            cy=float(line[7])/scale_y
            camLine[id]=f"{fx:.3f},{fy:.3f},{cx:.1f},{cy:.1f}"
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/back/images.txt",'r') as f:
        lines = f.readlines()
        for line in lines:
            if ".jpg" not in line:
                continue
            line=line.rstrip('\n').split(" ")
            id=line[0]
            imgName=line[-1]
            imgMap[imgName].append(camLine[id])

    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/nerf_24_exper/地面imgs文件已转换.txt",'r') as f:
        lines=f.readlines()
        for line in lines:
            imgName,rt=line.rstrip('\n').split(' ',1)
            rt=rt.replace(' ',', ')
            imgMap[imgName].append(rt)
        
    print("")


def processCamAndImg():
    from collections import defaultdict
    imgLine={}
    camLine={}
    imgMap={}
    # 最终的分辨率
    width,heigth=640,480

    # 处理cam文件
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/back/cameras.txt",'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            line=line.rstrip('\n').split(' ',1)
            id=line[0]
            back=line[1]
            camLine[id]=back
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/back/images.txt",'r') as f:
        lines = f.readlines()
        for line in lines:
            if ".jpg" not in line:
                continue
            line=line.rstrip('\n').split(" ")
            id=line[0]
            imgName=line[-1]
            imgMap[imgName]=camLine[id]
    
    list=[]
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/back/pairs-query-netvlad.txt",'r') as f:
        lines= f.readlines()
        for line in lines:
            line=line.split(' ')[0].split('/')[1]
            list.append(line)

    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/back/query_list_with_intrinsics.txt",'w') as f:
        for name,back in imgMap.items():
            if name in list:
                line=f"query/{name} {back}\n"
                f.write(line)
    with open("/home/lys/Workplace/python/SuperGluePretrainedNetwork/back/list_test.txt",'w') as f:
        for name,back in imgMap.items():
            if name in list:
                line=f"query/{name}\n"
                f.write(line)
    print("")

if __name__ == '__main__':
    # processImgandCam()
    #processCamAndImg()
     returnimgMap()
    # groundcord2aircord()
    # world2img()
    # img2world()