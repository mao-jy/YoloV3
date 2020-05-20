import os
import numpy as np


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename, target_filename):
        self.cluster_number = cluster_number
        self.filename = filename
        self.target_filename = target_filename

    def iou(self, boxes, clusters):
        """计算联合iou

        Args:
            boxes: [num_bboxes, [width, height]]
            clusters: [num_anchors, [width, height]]

        Returns:
            result: [num_bboxes, num_anchors]. result[i][j]表示第i个bbox和第j个anchor的iou
        """
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]        # [num_bboxes]
        box_area = box_area.repeat(k)               # [num_bboxes * num_anchors]. 注意和np.tile区别
        box_area = np.reshape(box_area, (n, k))     # [num_bboxes, num_anchors]. 第i行是将第i个bboxes的面积重复num_anchors次

        cluster_area = clusters[:, 0] * clusters[:, 1]      # [num_anchors]
        cluster_area = np.tile(cluster_area, [1, n])        # [num_anchors * num_bboxes]. 注意和repeat区别
        cluster_area = np.reshape(cluster_area, (n, k))     # [num_bboxes, num_anchors]. 第i行是num_anchors个anchors的面积

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)

        # [num_bboxes, num_anchors]. inter_area[i][j]的值是第i个bbox和第j个anchor的相交面积
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)

        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        """

        Args:
            boxes: [num_bboxes, [width, height]]
            k: 最终需要的anchors的个数
            dist: np.median返回中位数

        Returns:
            [num_anchors, [width, height]]
        """
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(      # clusters: [9, [width, height]].
            box_number, k, replace=False)]      # 从boxes中随机不重复取num_anchors个bbox作为初始anchors
        while True:
            # [num_bboxes, num_anchors]. distance[i][j]表示第i个bbox和第j个anchor的距离
            distances = 1 - self.iou(boxes, clusters)

            # [num_bboxes]. 与某个bboxes距离距离最近的anchor下标注
            current_nearest = np.argmin(distances, axis=1)

            if (last_nearest == current_nearest).all():
                break

            # 用多个bboxes的中位数更新anchors
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        """
        将聚类得到的anchors写入txt文件中
        """
        f = open(os.path.join(os.path.dirname(__file__), self.target_filename), 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        """将txt文件中的所有bbox的宽高提取出来

        Returns:
            result: [num_bboxes, [width, height]]
        """
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:                  # 每一行对应一张图片
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):  # 每行中多个bbox以空格分开，每个bbox的四个坐标值(xmin, ymin, xmax, ymax)以逗号分开
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        """
        对filename中的框进行聚类，得到k个anchors，并将结果写入新的target_filename中
        """
        all_boxes = self.txt2boxes()                            # [num_bboxes, [width, height]]
        result = self.kmeans(all_boxes, k=self.cluster_number)  # [num_anchors, [width, height]]
        result = result[np.lexsort(result.T[0, None])]          # 将result按宽度进行升序排列
        self.result2txt(result)

        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "2007_train.txt"
    target_filename = "model_data/train_anchors.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename, target_filename)
    kmeans.txt2clusters()
