from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

flags.DEFINE_string('dataset', '/home/j_m/Desktop/data_yolo/dataset/val/*.tfrecord', 'path to dataset')
flags.DEFINE_string('val_dataset', '/home/j_m/Desktop/data_yolo/dataset/val/*.tfrecord', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', '')
flags.DEFINE_string('classes', '/home/j_m/Desktop/data_yolo/dataset/classes.names', 'path to classes file')
flags.DEFINE_enum('mode', 'eager_tf', ['eager_fit', 'eager_tf', 'fit'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune', 'no_output_no_freeze', 'continue'],
                  'none: Training from scratch, '                  
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'no_output_no_freeze: Transfer all but output and freeze nothing, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only,'
                  'continue: Transfer all and freeze nothing')
flags.DEFINE_integer('size', 416, 'image size')     # 32的倍数
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
flags.DEFINE_integer('num_classes', 3, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', 80, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')


def main(_argv):
    # GPU设置
    physical_devices = tf.config.experimental.list_physical_devices('GPU')    # 获取所有物理GPU
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)       # 打开内存增长

    # 模型初始化
    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # 加载数据集
    # 训练集
    train_dataset = dataset.load_fake_dataset()
    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(    # 这里也做了resize，和后面的resize调用了一样的函数，后面的
            FLAGS.dataset, FLAGS.classes, FLAGS.size)     # resize应该删除，且这里的resize应该修改为不让图片失真的resize
    num_of_data = 0
    for _ in train_dataset:
        num_of_data += 1
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),          # 做resize和像素值小数化，这里的具体做法可能需要修改
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # 验证集
    val_dataset = dataset.load_fake_dataset()
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    # 迁移学习
    if FLAGS.transfer == 'none':                            # 不进行迁移学习
        pass
    elif FLAGS.transfer in ['darknet', 'no_output',
                            'no_output_no_freeze']:         # 只迁移某些层的参数并将这些层冻结
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':                         # 加载darknet层的参数并冻结
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        elif FLAGS.transfer == 'no_output':                     # 加载除输出层以外的参数并冻结
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(l.name).get_weights())
                    freeze_all(l)
        else:                                                   # 加载除输出层以外的参数并且不冻结
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(l.name).get_weights())

    else:                                                   # 迁移整个网络的所有参数并冻结某些层
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':                       # 迁移整个网络所有参数并冻结yolo_darknet
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':                        # 迁移整个网络所有参数并冻结所有参数
            freeze_all(model)
        elif FLAGS.transfer == 'continue':                      # 迁移整个网络进行训练
            pass

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)

    # loss: [高层loss, 中层loss函数, 低层loss]
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    if FLAGS.mode == 'eager_tf':

        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

        # 用于写日志文件
        num_of_batch = int(np.ceil(num_of_data / FLAGS.batch_size))
        logging.info("num of data: {}, batch size: {}, num of batch: {}".format(
                num_of_data, FLAGS.batch_size, num_of_batch))
        train_summary_writer = tf.summary.create_file_writer('logs/train')

        for epoch in range(FLAGS.epochs):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:

                    # 正则化损失
                    regularization_loss = tf.reduce_sum(model.losses)

                    # 预测损失
                    # outputs: (高层output, 中层output, 底层output)，格式为：
                    #   ((batch_size, h2, w2, num_anchors0, num_class+5),
                    #    (batch_size, h2*2, w2*2, num_anchors1, num_class+5)
                    #    (batch_size, h2*4, w2*4, num_anchors2, num_class+5))
                    # labels: 由GT框得到的target，格式为：
                    #   (高层target, 中层target, 低层target)
                    #   (batch_size, grid_size, grid_size, num_anchors, [x1, y1, x2, y2, 1, class])
                    # loss: [高层loss, 中层loss函数, 低层loss]
                    #     loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
                    #             for mask in anchor_masks]
                    # pred_loss: [(batch,), (batch,), (batch,)]
                    pred_loss = []
                    outputs = model(images, training=True)
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    loss_without_reg = tf.reduce_sum(pred_loss)

                    # 带正则项的损失
                    total_loss = loss_without_reg + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                logging.info("epoch_{}_batch_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

                # 每个batch记录一次训练集上的loss
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_without_reg.numpy(), step=(epoch * num_of_batch + batch))

                # 定期保存checkpoint
                    model.save_weights('checkpoints/yolov3_{}_{}.tf'.format(epoch, batch))

                # 定期计算mAP

            avg_loss.reset_states()
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True, save_freq=500),
            TensorBoard(log_dir='logs', update_freq=10)
        ]

        # history.history是一个字典，存放着训练过程的loss和其他metrics
        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
