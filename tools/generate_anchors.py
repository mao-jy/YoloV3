# 踩坑后的经验：
# 使用kmeans聚类获取anchors之前需要将初始的图片进行和算法中相同的预处理
# 常见的图片预处理方法是：pad到正方形+resize到416*416，所以对于初始的图片，也需要进行pad+resize，再进行框聚类
# 由于代码中是对归一化后的宽高做kmeans，代码中的宽高[width, height]是框的实际宽高[w, h]占图片宽高[img_width, img_height]的比例
# 即[width, height] = [w, h] / [img_width, img_height]，所以仅需考虑pad这一步对[width, height]带来的影响

import os
import cv2
import numpy as np
from tqdm import tqdm


def iou(box, clusters):
    """计算单个box和所有clusters的iou

    Args:
        box: [width, height]
        clusters: [num_bboxs, [width, height]]

    Returns:
        [num_bboxs,,]. 单个bbox和所有clusters的iou
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """计算平均iou

    Args:
        boxes: [num_bboxes, [width, height]]
        clusters: [num_clusters, [wisth, height]]

    Returns:
        float.
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """kmeans生成聚类中心

    Args:
        boxes: [num_bboxes, [width, height]]
        k: 聚类的个数
        dist: 更新聚类中心的方法

    Returns:
        [num_clusters, [width, height]]
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))         # [num_bboxes, num_clusters]
    last_clusters = np.zeros((rows,))       # [num_bboxes,]

    np.random.seed()                        # 随机数种子

    # [num_bboxes, num_clusters]. 从bboxes中随机不重复取num_bboxes个[width, height]
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:

        # distances: [num_bboxes, num_clusters]. 所有anchors和所有bboxes的distance
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        # [num_bboxes,]. 和每个bboxes距离最近的是哪个cluster
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # 取得所有和当前cluster距离最近的bboxes的中位数作为新的聚类中心
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def get_wh(txt_path, anno_dir, img_dir):
    """获取所有框归一化形式的w, h

    Args:
        txt_path: 训练集文件名
        anno_dir: txt标注文件所在目录
        img_dir: 图片目录

    Returns:
        [num_bboxes, [width, height]]. width, height的值介于0, 1之间
    """

    with open(txt_path, 'r') as f:
        all_filenames = f.read().splitlines()

    bboxes = []

    for filename in tqdm(all_filenames):

        filename = filename.split(r'/')[-1].replace('.jpg', '.txt')
        txt_anno_path = os.path.join(anno_dir, filename)
        img_path = os.path.join(img_dir, filename.replace('.txt', '.jpg'))

        with open(txt_anno_path, 'r') as f:

            lines = f.read().splitlines()

            for line in lines:
                line = line.split(' ')
                width, height = float(line[3]), float(line[4])

                # 算法中的数据预处理时会先将图片pad成正方形，然后resize到416*416，框的聚类前也要进行一样的处理
                # 这一步相当于计算框在pad后的图片上的宽、高
                img = cv2.imread(img_path)
                img_height, img_width = img.shape[0], img.shape[1]
                if img_height > img_width:
                    width = width * img_width / img_height
                elif img_height < img_width:
                    height = height * img_height / img_width

                bboxes.append([width, height])

    return np.array(bboxes)


def main():
    txt_path = r'C:\Users\J_M\Desktop\anchors_get\train.txt'  # 训练集文件名
    anno_dir = r'C:\Users\J_M\Desktop\anchors_get\trainval\labels'  # txt标注目录
    img_dir = r'C:\Users\J_M\Desktop\anchors_get\trainval\images'  # 图片目录
    num_clusters = 9

    # 从标注文件中加载所有框的w和h，w和h的值需要是归一化后的形式
    print('loading bboxes whdth and height...')
    bboxes_wh = get_wh(txt_path, anno_dir, img_dir)
    print('bboxes width and height loaded\n')

    # kmeans生成聚类中心
    print('generating clusters...')
    clusters = kmeans(bboxes_wh, num_clusters)
    print('cluster:\n', clusters, '\n')

    # 聚类框
    anchors = np.round(clusters * 416)
    print('anchors:\n', anchors, '\n')

    # 计算平均iou
    print("Average IoU: {:.2f}%".format(avg_iou(bboxes_wh, clusters) * 100))
    yolov3clusters = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    yolov3clusters = np.array(yolov3clusters) / 416.0
    print("Average IoU with VOC clusters: {:.2f}%\n".format(avg_iou(bboxes_wh, yolov3clusters) * 100))

    ratios = np.around(clusters[:, 0] / clusters[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))


if __name__ == "__main__":
    main()
