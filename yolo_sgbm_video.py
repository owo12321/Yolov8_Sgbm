import cv2
import numpy as np
import time
import random
import math


import sys
import shutil,os
sys.path.append('yolov8_pytorch_master')
from yolov8.yolo_sgbm import YOLO

yolo = YOLO()


# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
left_camera_matrix = np.array(  [[1.06265426e+03, 0.00000000e+00, 6.40549410e+02],
                                [0.00000000e+00, 1.06309321e+03, 4.90735489e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

right_camera_matrix = np.array( [[1.06520238e+03, 0.00000000e+00, 6.14658394e+02],
                                [0.00000000e+00, 1.06445423e+03, 4.80741249e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[-0.01085743, 0.17822576, -0.00781658, 0.00246898, -0.1378341 ]])
right_distortion = np.array([[-0.00601122, 0.19590039, -0.0077834, 0.00157652, -0.16437705]])

# 旋转矩阵
R = np.array(   [[ 9.99910187e-01, 1.78596734e-04, 1.34009873e-02],
                [-2.38730178e-04, 9.99989910e-01, 4.48577430e-03],
                [-1.34000510e-02, -4.48857064e-03, 9.99900141e-01]])
# 平移矩阵
T = np.array([-64.65432992,-0.2980873,3.76597753])


# size = (640, 480)
size = (1280, 720)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
# print(Q)


# 加载视频文件
video_path = './input/outdoor.mp4'
filename = video_path.split('/')[-1][:-4]

# 创建保存文件夹
if os.path.exists(f'./output/{filename}'):
    shutil.rmtree(f'./output/{filename}')
os.mkdir(f'./output/{filename}')

# cv2读取视频流
capture = cv2.VideoCapture(video_path)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[1]*2)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[0])

# 原视频帧率
vid_fps = int(capture.get(cv2.CAP_PROP_FPS))
# 原视频总帧数
total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

# 输出流
main_writer = cv2.VideoWriter(f'./output/{filename}/{filename}_main.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, (size[0], size[1]))
dis_color_writer = cv2.VideoWriter(f'./output/{filename}/{filename}_dis_color.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, (size[0], size[1]))
dis_gray_writer = cv2.VideoWriter(f'./output/{filename}/{filename}_dis_gray.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, (size[0], size[1]))



WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

# 读取视频
fps = 0.0
frame_idx = 1
while True:
    # 开始计时
    t1 = time.time()
    # 是否读取到了帧，读取到了则为True
    ret, frame = capture.read()
    if not ret:
        break
    
    # 切割为左右两张图片
    frame1 = frame[0:size[1], 0:size[0]]
    frame2 = frame[0:size[1], size[0]:size[0]*2]

    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # ------------------------------------------------------------------------------------------------------
    blockSize = 3
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)

    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16

    result_list = yolo.detect_image(frame1)
    for i, rectangle in enumerate(result_list):
        x, y = (rectangle[0]+rectangle[2])//2, (rectangle[1]+rectangle[3])//2
        world_x, world_y, world_z = threeD[y][x][0]/1000.0, threeD[y][x][1]/1000.0, threeD[y][x][2]/1000.0
        distance = math.sqrt(world_x ** 2 + world_y ** 2 + world_z ** 2)

        color = (0,255,0) if distance>5 else (0,0,255)
        cv2.rectangle(frame1, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), color, 2)
        cv2.putText(frame1, rectangle[4], (rectangle[0], rectangle[1]), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        cv2.putText(frame1, f'distance = {distance:.1f}', (rectangle[0], rectangle[3]-3), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

    #完成计时，计算帧率
    fps = (fps + (1. / (time.time() - t1))) / 2
    print(f'frame: {frame_idx}/{total_frame}, fps: {fps:.3f}')
    frame_idx += 1
    frame = cv2.putText(frame, "fps= %.3f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("depth", dis_color)
    dis_color_writer.write(dis_color)

    cv2.imshow(WIN_NAME, disp)  # 显示深度图的双目画面
    dis_gray_writer.write(cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR))   # 灰度转BGR再保存，不然只有一个通道
    
    cv2.imshow("left", frame1)
    main_writer.write(frame1)

    # 若键盘按下q则退出播放
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


# 释放资源
capture.release()
main_writer.release()
dis_color_writer.release()
dis_gray_writer.release()

# 关闭所有窗口
cv2.destroyAllWindows()