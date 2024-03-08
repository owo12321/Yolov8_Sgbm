# Yolov8_Sgbm
结合yolov8和sgbm，实现检测物体并测量距离  
![image](./src/outdoor_main.gif)  

## 1 环境
```
pip install -r requirements.txt
```
在Python3.8、cuda10.2下能够正常运行，其他版本请自行测试  

将yolov8文件放在Yolov8_Sgbm/yolov8_pytorch文件夹下（yolo.py和yolo_sgbm.py在同一层级）：https://github.com/bubbliiiing/yolov8-pytorch  

将权值文件放在Yolov8_Sgbm/yolov8_pytorch/model_data文件夹下：https://github.com/bubbliiiing/yolov8-pytorch?tab=readme-ov-file#%E6%96%87%E4%BB%B6%E4%B8%8B%E8%BD%BD  

## 2 运行
yolo_sgbm_video.py里，填写相机的内参矩阵、畸变系数、旋转矩阵和平移矩阵填写输入视频的路径，然后运行yolo_sgbm_video.py  
输出结果存放在Yolov8_Sgbm/output文件夹下  

## 3 参考
https://www.bilibili.com/video/BV1zT411w7oZ  
https://github.com/yzfzzz/Stereo-Detection  
https://github.com/bubbliiiing/yolov8-pytorch  
