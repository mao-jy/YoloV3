# note: coco格式数据集转换成tfrecord格式数据集

import os
import json
import tqdm
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('image_dir', '/home/j_m/Desktop/train/images', 'images directory')
flags.DEFINE_string('anno_file', '/home/j_m/Desktop/train/train.json', 'annotation file path')
flags.DEFINE_string('output_prefix', '/home/j_m/Desktop/train/train_', 'prefix of output tfrecord name')
# 所有数据生成tfrecord可能过大，需要存放成多个tfrecord


def build_single(annotation):
    """
    构建单张图片的tfrecord
    """

    img_path = os.path.join(FLAGS.image_dir, annotation['filename'])
    img_raw = open(img_path, 'rb').read()

    height, width = annotation['height'], annotation['width']

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []

    for idx in range(len(annotation['bboxes'])):
        bbox = annotation['bboxes'][idx]
        xmin.append(float(bbox[0]) / width)
        ymin.append(float(bbox[1]) / height)
        xmax.append(float(bbox[0] + bbox[2]) / width)
        ymax.append(float(bbox[1] + bbox[3]) / height)
        classes_text.append(annotation['labels'][idx].encode('utf8'))

    single_tfrecord = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
    }))

    return single_tfrecord


def main(_argv):

    with open(FLAGS.anno_file, 'r', encoding='utf8') as f:
        json_dict = json.load(f)
    logging.info('Json file loaded.')

    id2label = {}
    for cat in json_dict['categories']:
        id2label[cat['id']] = cat['name']
    logging.info("Id2label parse finished. %s", id2label)

    images = json_dict['images']

    logging.info("Start to build tfrecord...")
    for batch in range(len(images) // 4000 + 1):
        writer = tf.io.TFRecordWriter(FLAGS.output_prefix + str(batch + 1) + '.tfrecord')
        start = 4000 * batch
        end = min(4000 * (batch + 1), len(images))
        for item in tqdm.tqdm(images[start:end]):

            image_id, filename, height, width = item['id'], item['file_name'], item['height'], item['width']

            bboxes = []
            labels = []
            for anno in json_dict['annotations']:
                if anno['image_id'] == image_id:
                    bboxes.append(anno['bbox'])
                    labels.append(id2label[anno['category_id']])

            if len(bboxes) == 0:     # 过滤无标签的图片
                continue

            annotation = {'filename': filename,
                          'height': height,
                          'width': width,
                          'bboxes': bboxes,
                          'labels': labels}

            single_tfrecord = build_single(annotation)
            writer.write(single_tfrecord.SerializeToString())

        writer.close()
        logging.info('batch {}/{} finished.'.format(batch + 1, len(images) // 4000 + 1))
    logging.info('Tfrecord built.')


if __name__ == '__main__':
    app.run(main)
