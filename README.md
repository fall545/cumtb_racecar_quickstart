# 🚗 cumtb_racecar_quickstart

**中国矿业大学（北京）智能车校赛视觉方案 · 快速上手参考**

本仓库提供一个轻量级视觉处理示例，可用于智能车校赛视觉算法的调试、图像预处理与线检测等任务。基于 Python + OpenCV + Scikit-learn，适合快速实验与算法验证。

---

## 📦 环境配置

### 1. 创建 Conda 环境（可选）

```bash
conda create -n vision python=3.12
conda activate vision
```

### 2. 安装依赖

```bash
pip install scikit-learn
pip install opencv-python
```

---

## 🏁 如何运行

### 编辑脚本中的图片路径

打开 `extract.py`，找到：

```python
img = "your_image_path_here"
```

替换为你的输入图像路径。

### 然后运行：

```bash
python extract.py
```

---

## 📸 图像处理示例

下方展示基于该代码的输入与输出结果。

### 🔹 输入图像

| 输入一                           | 输入二                           |
| ----------------------------- | ----------------------------- |
| ![input1](./photo/in1.png) | ![input2](./photo/in2.png) |

### 🔹 处理后输出

| 输出一                             | 输出二                             |
| ------------------------------- | ------------------------------- |
| ![output1](./photo/out1.png) | ![output2](./photo/out2.png) |

---

🔧 算法工作原理

本视觉方案基于赛道区域提取 + 多路径黑色像素扫描 + 贝塞尔拟合实现赛道中心提取，最终输出误差用于车辆控制。

第一步：赛道区域提取（图像二值化）

将图像转为灰度图（Grayscale）

使用阈值分割提取黑色赛道区域（Binary Threshold）

形成黑白二值图，黑色为赛道

第二步：底部聚类扫描（起点生成）

从图像底部开始，连续取 n 层像素形成扫描矩形窗口

使用聚类（如 K-Means 或 Connected Components）合并相邻黑色区域

得到赛道的底部中心点作为初始路径点

（Python OpenCV 示例：cv2.connectedComponents、cv2.findContours）

第三步：基于 BFS 的路径拓展（多路径追踪）

以当前点为圆心，从 θ = 0° 到 180° 范围、扫描距离为 dis

BFS 搜索黑色赛道像素

对扫描内的黑色区域再次聚类 → 得到新赛道点

若找到多个点，则路径分裂（分支搜索）

将所有路径点 push 进列表

当路径长度超过 threshold 停止

第四步：贝塞尔曲线拟合（路径平滑）

从扫描获得的曲线点集进行拟合

使用二次或三次贝塞尔进行平滑

从贝塞尔曲线取 k 个点

按从下到上加权（递减方式）求和

得到最终目标横向偏移 target

第五步：误差计算与控制输出

error = target - 图像中心


## 📂 文件结构

```
cumtb_racecar_quickstart/
│
├── extract.py
├── photo/
│   ├── in1.png
│   ├── in2.png
│   ├── out1.png
│   ├── out2.png
│
└── README.md
```


## 📝 校赛加油呀

