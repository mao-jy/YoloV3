from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .utils import broadcast_iou

flags.DEFINE_integer('yolo_max_boxes', 100, 'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:            # 只改变通道数,大小不变
        padding = 'same'
    else:                       # 尺寸变化公式: (height_or_width + 1 - size) // stride    +   1
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):           # 类似于残差网络的identity block,(h, w, c) -> (h, w, filters)
    prev = x
    x = DarknetConv(x, filters // 2, 1)    # strides默认为1,只改变通道数,宽高不变
    x = DarknetConv(x, filters, 3)         # strides默认为1,只改变通道数,宽高不变
    x = Add()([prev, x])                   # 保证了输入输出完全相同
    return x


def DarknetBlock(x, filters, blocks):           # 整个DarknetBlock的尺寸变化为:(h, w, c) -> (h//2, w//2, filters)
    x = DarknetConv(x, filters, 3, strides=2)   # (h, w) -> (h//2, w//2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)         # 类似于残差网络的identity block,输入输出宽高和通道数完全相同
    return x


def Darknet(name=None):                    # darknet-53由52个卷积层和一个全连接层组成,下面为用于特征提取的52层
    x = inputs = Input([None, None, 3])    # (h, w, 3)
    x = DarknetConv(x, 32, 3)              # (h, w, 32),含有卷积层:1
    x = DarknetBlock(x, 64, 1)             # (h//2, w//2, 64),含有卷积层:1+2*1=3
    x = DarknetBlock(x, 128, 2)            # (h//2//2, w//2//2, 128),含有卷积层:1+2*2=5
    x = x_36 = DarknetBlock(x, 256, 8)     # (h//2//2//2, w//2//2//2, 256),含有卷积层:1+2*8=17
    x = x_61 = DarknetBlock(x, 512, 8)     # (h//2//2//2//2, w//2//2//2//2, 512),含有卷积层:1+2*8=17
    x = DarknetBlock(x, 1024, 4)           # (h//2//2//2//2//2, w//2//2//2//2//2, 1024),含有卷积层:1+2*4=9
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def DarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def YoloConv(filters, name=None):
    # YoloCov1: (h, w, c) -> (h, w, filters), YoloCov1: (h, w, c) -> (h*2, w*2, filters)
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):         # YoloConv2, YoloConv3
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            x = DarknetConv(x, filters, 1)  # strides默认为1,只改变通道数,宽高不变
            x = UpSampling2D(2)(x)          # 上采样,宽高变成两倍
            x = Concatenate()([x, x_skip])  # 通道数上的堆叠
        else:                               # YoloConv1
            x = inputs = Input(x_in.shape[1:])

        # 5层DarknetConv: strides默认为1,宽高不变,通道数最终变为filters
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)     # strides默认为1,宽高不变,只改变通道数

        # strides默认为1,宽高不变,只改变通道数
        # 通道数变为: 每个点先验框的数量 * ( 类别数(4) + 偏移量(4) +置信度(1) )
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)

        # x: (..., h, w, num_anchors, num_classes+5)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output


def yolo_boxes(pred, anchors, classes):
    """对yolo网络的输出进行处理，便于计算loss和输出结果

    Args:
        pred: [batch_size, grid_size, grid_size, num_anchors, num_classes+5]
        anchors: 当前特征层上的所有anchors
        classes: 类别数

    Returns:
        bbox: (batch_size, grid_size, grid_size, num_anchors, (x1, y1, x2, y2)). 用于计算iou和输出结果
        objectness: (batch_size, grid_size, grid_size, num_anchors, confidence). 置信度
        class_probs: (batch_size, grid_size, grid_size, num_anchors, 80). 80个类的预测结果
        pred_box: (batch_size, grid_size, grid_size, num_anchors, (dx, dy, dw, dh)). 用于计算loss
    """
    # pred: (batch_size, grid_size, grid_size, anchors, num_classes+5))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    # 中心坐标，置信度，80个类的预测结果最后使用sigmoid激活函数
    box_xy = tf.sigmoid(box_xy)                      # (batch_size, grid_size, grid_size, num_anchors, (dx, dy))
    objectness = tf.sigmoid(objectness)              # (batch_size, grid_size, grid_size, num_anchors, confidence)
    class_probs = tf.sigmoid(class_probs)            # (batch_size, grid_size, grid_size, num_anchors, 80个类的预测结果)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # (batch_size, grid_size, grid_size, num_anchors, (dx, dy, dw, dh))

    # tf.meshgrid -> ((grid_size, grid_size), (grid_size, grid_size)) 用两个列表生成网格
    # tf.stack -> (grid_size, grid_size, 2)
    # tf.expand_dims -> (grid_size, grid_size, 1, 2)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    # box_xy: (batch_size, grid_size, grid_size, anchors, (x, y)) 中心点坐标变换，值介于0,1之间
    # box_wh: (batch_size, grid_size, grid_size, anchors, (w, h)) 宽高变换，值介于0,1之间
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    # bbox: (batch_size, grid_size, grid_size, anchors, (x1, y1, x2, y2))
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    # x_36: (h//2//2//2, w//2//2//2, 256)
    # x_61: (h//2//2//2//2, w//2//2//2//2, 512)
    # x: (h//2//2//2//2//2, w//2//2//2//2//2, 1024) = (h2, w2, 1024)
    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    # x: (h2, w2, 512)
    # output0: (h2, w2, num_anchors0, num_class+5)
    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    # x: (h2*2, w2*2, 256)
    # output1: (h2*2, w2*2, num_anchors1, num_class+5)
    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    # x: (h2*4, w2*4, 128)
    # output2: (h2*4, w2*4, num_anchors2, num_class+5)
    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    # anchors: 单个特征层上的所有anchors
    # ignore_thres: 置信度阈值
    def yolo_loss(y_true, y_pred):
        """计算单个特征层上的损失

        Args:
            y_true: (batch_size, grid_size, grid_size, num_anchors, [x1, y1, x2, y2, 1, class])
            y_pred: (batch_size, h, w, num_anchors, num_class+5)
                其中h = w = grid_size，函数中的注释全部用grid_size来表示
                于是两个变量的尺寸如下：
                    y_true: (batch_size, grid_size, grid_size, num_anchors, (x1, y1, x2, y2, 1, class))
                    y_pred: (batch_size, grid_size, grid_size, num_anchors, num_classes+5)
        Returns:
            total_loss: (batch_size)
        """
        # 1. 对于yolo网络的输出进行处理
        # pred_box: (batch_size, grid_size, grid_size, num_anchors, (x1, y1, x2, y2)). 用于输出
        # pred_obj: (batch_size, grid_size, grid_size, num_anchors, confidence). 置信度
        # pred_class: (batch_size, grid_size, grid_size, num_anchors, 80). 80个类的预测结果
        # pred_xywh: (batch_size, grid_size, grid_size, num_anchors, (dx, dy, dw, dh)). 用于计算loss
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]   # (batch_size, grid_size, grid_size, num_anchors, (x, y))
        pred_wh = pred_xywh[..., 2:4]   # (batch_size, grid_size, grid_size, num_anchors, (w, h))

        # 2. 对于GT框的格式进行改变
        # true_box: (batch_size, grid_size, grid_size, num_anchors, (x1, y1, x2, y2))
        # true_obj: (batch_size, grid_size, grid_size, num_anchors, 1). GT框的置信度，值全为1
        # true_class_idx: (batch_size, grid_size, grid_size, num_anchors, 1). GT框的类别号
        # true_xy: (batch_size, grid_size, grid_size, num_anchors, (x, y))
        # true_wh: (batch_size, grid_size, grid_size, num_anchors, (w, h))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # 对于小的框给予更大权重
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. 讲GT框转换成dx, dy, dw, dh格式
        # (grid_size, grid_size, 1, 2)
        # true_xy: (batch_size, grid_size, grid_size, num_anchors, (dx, dy))
        # true_wh: (batch_size, grid_size, grid_size, num_anchors, (dw, dh))
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. 计算mask
        # obj_mask: (batch_size, grid_size, grid_size, num_anchors). GT框的置信度，有框的地方值全部1
        obj_mask = tf.squeeze(true_obj, -1)

        # pred_box: (batch_size, grid_size, grid_size, num_anchors, (x1, y1, x2, y2))
        # true_box: (batch_size, grid_size, grid_size, num_anchors, (x1, y1, x2, y2))
        # obj_mask: (batch_size, grid_size, grid_size, num_anchors). GT框的置信度，有框的地方值全部1
        # tmp1 = tf.boolean_mask(x[1], tf.cast(x[2], tf.bool)) -> (num_gt_boxes, (x1, y1, x2, y2))
        # tmp2 = broadcast_iou(x[0], tmp1) -> (grid_size, grid_size, num_anchors, num_gt_boxes)
        # tmp3 = tf.reduce_max(tmp2, axis=-1) -> (grid_size, grid_size, num_anchors) 每个anchor对应的最大iou
        # best_iou: (batch_size, grid_size, grid_size, num_anchors)
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. 计算损失
        # 坐标损失，对于置信度大于阈值的预测框，计算dx, dy, dw, dh的损失，且越小的框损失权重越大
        # (batch_size, grid_size, grid_size, num_anchors)
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

        # 置信度损失
        # (batch_size, grid_size, grid_size, num_anchors)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        # obj_loss = obj_mask * obj_loss + (1 - obj_mask) * (1 - ignore_mask) * obj_loss
        # obj_loss = obj_mask * obj_loss + lambda * (1 - obj_mask) * obj_loss  按照公式理解

        # 对于有物体的地方，计算分类损失
        # (batch_size, grid_size, grid_size, num_anchors)
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        # 6. 计算损失和
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))            # (batch_size)
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))            # (batch_size)
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))          # (batch_size)
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))      # (batch_size)

        return xy_loss + wh_loss + obj_loss + class_loss            # (batch_size)
    return yolo_loss

