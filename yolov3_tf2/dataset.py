import tensorflow as tf
from absl.flags import FLAGS


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    """在某个特征层上，对GT框进行格式转换，方便后续计算loss

    Args:
        y_true: (batch_size, num_boxes, [x1, y1, x2, y2, class, best_anchor]).
            坐标是小数形式，class类别号,best_anchor是与当前GT框iou最大的anchors
        grid_size: 当前特征层的宽高为 grid_size * grid_size
        anchor_idxs: 当前特征层需要使用哪几个anchors
            值得注意的是，特征层的宽高越大，其对应的anchors应该越小；特征层宽高越小，其对应的anchors应该越大
            即：低层特征具有更好的细节特征，有利于提取小物体；高层特征具有更好的语义特征，更加有利于提取大物体

    Returns:
        (batch_size, grid_size, grid_size, num_anchors, 6)
            其中num_anchors表示与当前特征层有关的的anchor数，值为tf.shape(anchor_idxs)[0]
            其中6的内容为：(x1, y1, x2, y2, 1, class)
            1表示置信度，由于是GT框，所以置信度为1，class为类别号，坐标值介于0,1之间
    """
    N = tf.shape(y_true)[0]

    # y_true_out: (batch_size, grid_size, grid_size, num_anchors, 6)
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

    idx = 0
    for i in tf.range(N):                          # 遍历batch中的每一张图片
        for j in tf.range(tf.shape(y_true)[1]):    # 遍历当前图片中的每一个GT框

            if tf.equal(y_true[i][j][2], 0):
                continue

            # anchor_eq: (num_anchor_idxs) bool. 当前GT框对应的anchor是否与该特征层中的某个anchors对应
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):    # 如果能找到
                box = y_true[i][j][0:4]

                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2    # GT框的中心点坐标(坐标值介于0,1之间)

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)     # GT框对应当前特征层的第几个anchor
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)  # 找到中心点坐标在grid上对应哪个点(整数值)

                # idx: 整个batch中在这个特征层上找到的框计数
                # i: 第几张图片
                # grid_xy[1]: y坐标 int.
                # grid_xy[0]: x坐标 int. GT框的中心点对应当前特征层上的哪个像素点
                # anchor_idx int. GT框对应哪个anchor
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])

                # idx: 整个batch中在这个特征层上找到的框计数
                # box[0:4]: [x1, y1, x2, y2]. GT框在原图上的坐标，坐标值介于0,1之间
                # y_true[i][j][4]: int. GT框的类别号
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])

                idx += 1

    # tf.tensor_scatter_nd_update: 对于y_true_out中indexes所指示的位置，用update中的内容进行更新
    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    """对三个特征层上的GT框进行格式转换

    Args:
        y_train: (batch_size, num_boxes, [x1, y1, x2, y2, class]). 坐标值介于0,1之间，class为类别号
        anchors: (num_all_anchors, [width, height]). 所有特征层上的所有anchors的宽高
        anchor_masks: (特征层数, 在此特征层上初始化哪几个anchor)
        size: int 图片尺寸为(size, size, 3)

    Returns:
        (高层特征层的target, 中间一层的特征层的target, 低层特征层的target)
            target的格式：(batch_size, grid_size, grid_size, num_anchors, 6)
            高层grid_size小，框大；低层grid_size大，框小
            其中num_anchors表示由当前特征层确定的anchor数量
            其中6的内容为：[x1, y1, x2, y2, 1, class]
            1表示置信度，由于是GT框，所以置信度为1，class为类别号，坐标值介于0,1之间
    """
    y_outs = []
    grid_size = size // 32

    # 获取anchors相关宽,高,面积
    # anchors: (num_anchors, 2)
    # anchors: (num_anchors)
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]

    # 获取GT框的宽,高,面积
    # box_wh: (batch_size, num_boxes, num_anchors, 2)
    # box_area: (batch_size, num_boxes, num_anchors)
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]

    # 计算所有GT框和所有anchors的iou
    # intersection: (batch_size, num_boxes, num_anchors)
    # iou: (batch_size, num_boxes, num_anchors)
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
                   tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)

    # anchor_idx: (batch_size, num_boxes, 1). 指示每一个GT框和哪个anchor的iou最大
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    # y_train: (batch_size, num_boxes, 6)
    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    """
    resize，并将坐标值化为0,1之间
    """
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


IMAGE_FEATURE_MAP = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, class_table, size):
    """解析tfrecord中的单张图片和标注

    Args:
        tfrecord: 单张图片的tfrecord
        class_table: 标签(str) -> 类别号(int)
        size: 将图片resize到(size, size, 3)

    Returns:
        x_train: (size, size, 3)
        y_train: (FLAGS.yolo_max_boxes, 5). 存放[x_min, y_min, xmax, ymax],坐标为小数形式
    """
    # 将单张图片的tfrecord按照IMAGE_FEATURE_MAP格式解析
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)  # 解析tfrecord文件

    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)  # 解码图片
    x_train = tf.image.resize(x_train, (size, size))  # 将图片resize成正方形

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)  # 标签(str) -> 类别号(int)

    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),  # GT框整合
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)  # GT框填充

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file, size=416):
    """
    加载所有tfrecord并进行解析
    """
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, -1, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)

    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('./example/girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
                 [0.18494931, 0.03049111, 0.9435849, 0.96302897, 0],
                 [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
                 [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
             ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
