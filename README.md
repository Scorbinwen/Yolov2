# Yolov2 Implementation
## 目的
学习Yolov2算法，以及探究深度学习算法调试手段

## 结果
| | training set | test set | mAP@416 | mAP@544 |
| :--: | :--: | :--: | :--: | :--: |
|this repo|VOC2007+2012|VOC2007|72.7|74.6|
|original paper|VOC2007+2012|VOC2007|76.8|78.6|
Running time: ~19ms (52FPS) on GTX 1080

## 训练Yolov2:
执行train.py --m
## 测试Yolov2:
执行eval.py
## 在tensorboard上查看损失函数和预测结果：
在命令行输入：
tensorboard --logdirs="logs" --port=6006
然后在本地浏览器中输入:
http://localhost:6006/即可查看
