## 网络结构
### 基本单元：Convolutional和Residual
Convolution：卷积层-BN层-激活层。

<img src="https://i.loli.net/2020/05/17/ta5WPwSHRcQ986v.png" height=70>

</br>
</br>
Residual：以Convolution为基础，且结构类似Resnet中的Identity Block。

<img src="https://i.loli.net/2020/05/17/K2CYwrPAfUpmhi1.png" height="140">

</br>
### Darknet-53
在上面两种基本单元的基础上，搭建Darknet-53。

<img src="https://i.loli.net/2020/05/17/OMkufShXAs9gPZe.png" height=600>

</br>
### YOLO
使用Darknet-53的前52个卷积层进行特征提取，再做类似于FPN的操作提取多层特征。不同之处在于：FPN将高层特征做上采样后与低层特征相加，而YOLO中是将高层特征上采样后与低层特征做concat操作。

<img src="https://i.loli.net/2020/05/17/o2qKreyIU8gd9DZ.png" height=800>

</br>
</br>
## Loss
由三部分构成：中心坐标和宽高损失，置信度损失，分类损失。三部分损失由&lambda;平衡。

<img src="https://i.loli.net/2020/05/17/NJcr9VI6pCMynxt.png" height=300>

由于论文中并没有给出具体的损失函数，所以不同的博客，代码对损失函数的解析可能并不相同，但是基本结构都是一样的。

</br>
### 中心坐标和宽高损失

<img src="https://i.loli.net/2020/05/17/ODfGSVgbiZlPIap.png" height=130>

上式的x,y,w,h并不是中心坐标和宽高值，而是它们的偏移量。在本次研究的代码中，上式中的&lambda;<sub>coord</sub>设置为1，而对于w和h并没有取根号，而是和x,y一样直接计算差的平方。对于不同大小的物体，大的物体会带来更大的坐标和宽高损失，要平衡大小物体的损失，所以基于GT框的大小给予不同的框权重，权重大小为2-w\*h。
```
# box_loss_scale即为2-w*h
xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
```

问题：框的大小与中心坐标损失应该是无关的，为什么中心坐标前面也要加上权重？

</br>
### 置信度损失

<img src="https://i.loli.net/2020/05/17/zV7nCfXkEQbargx.png" height=130>

置信度误差使用交叉熵进行计算。损失分为两部分：有物体，没有物体，其中没有物体损失部分还增加了权重系数。添加权重系数的原因是，对于一幅图像，一般而言大部分框是不包含待检测物体的，这样会导致没有物体的计算部分贡献会大于有物体的计算部分，进而导致网络倾向于预测单元格不含有物体。因此，我们要减少没有物体计算部分的贡献权重&lambda;<sub>noobj</sub>，比如取值为：0.5。

在本次研究的代码中，上述权重(即式中的&lambda;<sub>noobj</sub>)设置为1。对于有物体的位置，计算交叉熵，对于没有物体且IOU低于ignore_thres的位置，计算交叉熵，对两者求和。而忽略掉了那些没有物体且IOU大于0.5的预测框置信度损失，可能是因为nms会去掉这些框，所以不纳入损失计算。

</br>
### 分类损失

<img src="https://i.loli.net/2020/05/17/HNjDCRwtBS1qkld.png" height=70>

对于有物体的位置，计算交叉熵。分类之前的激活函数使用了sigmoid而非softmax，原因是：为了能够检测多标签物体的多个标签，防止softmax激活函数导致的不同类的相互抑制，如果不存在多标签物体，可以将激活函数替换为softmax。

</br>
</br>
## 一些细节
### 初始anchors的生成
对于YOLO网络输出的三个特征层，在它们上面分别定义不同大小的anchors,下图是YOLOv3作者通过在VOC数据集上进行聚类得到的anchors。因此对于自己的数据集，可以进行相同的聚类算法得到一组新的anchors，或许可以改善效果。

<img src="https://i.loli.net/2020/05/17/ON872bCtx35AusX.png" height=90>

</br>
### 图片预处理
本次研究的代码中，使用了resize直接将图片变成416\*416的正方形，引起了图片失真。这里可以修改为等比缩放后进行零填充，应该可以改善效果。
