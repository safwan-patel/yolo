import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Reshape, BatchNormalization, LeakyReLU
from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as ResNet50
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

OBJECT_SCALE = 5.0
NO_OBJECT_SCALE = 1.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0


class Yolo:

    def __init__(self, input_shape, labels, grid, anchors, max_grid_box=5, max_image_box=50, name='yolo'):
        self.input_shape = input_shape
        self.labels = labels
        self.class_weights = np.ones(len(self.labels), dtype='float32')
        self.grid = grid
        self.anchors = anchors
        self.max_grid_box = max_grid_box
        self.max_image_box = max_image_box
        self.name = name
        self.init_layers()
        self.build()

    def init_layers(self):

        grid_width, grid_height = self.grid
        self.input_image = Input(shape=self.input_shape)
        self.true_boxes = Input(shape=(1, 1, 1, self.max_image_box, 4))
        self.y_true = Input(shape=(grid_width, grid_height,
                                   self.max_grid_box, 4+1+len(self.labels)))

        self.resnet = ResNet50(
            include_top=False, weights='imagenet', input_shape=self.input_shape)

        self.res_conv1 = Conv2D(512, (1, 1), strides=(
            1, 1), padding='same', name='res_conv_1')
        self.norm1 = BatchNormalization(name='norm_1')
        self.activ1 = LeakyReLU(alpha=0.1)

        self.res_conv2 = Conv2D(self.max_grid_box * (4 + 1 + len(self.labels)),
                                (1, 1), strides=(1, 1), padding='same', name='res_conv_2')
        self.norm2 = BatchNormalization(name='norm_2')
        self.activ2 = LeakyReLU(alpha=0.1)

        self.reshape = Reshape(
            (grid_width, grid_height, self.max_grid_box, 4 + 1 + len(self.labels)))

    def build(self):

        x = self.resnet(self.input_image)

        x = self.res_conv1(x)
        x = self.norm1(x)
        x = self.activ1(x)

        x = self.res_conv2(x)
        x = self.norm2(x)
        x = self.activ2(x)

        self.y_predict = self.reshape(x)
        self.model = Model([self.input_image, self.y_true,
                            self.true_boxes], self.y_predict, name=self.name)

    def compile(self, optimizer=Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)):
        self.build()
        self.model.add_loss(self.custom_loss(
            self.y_true, self.y_predict, self.true_boxes))
        self.model.compile(optimizer=optimizer)

    def custom_loss(self, y_true, y_pred, true_boxes):
        mask_shape = tf.shape(y_true)[:4]
        grid_width, grid_height = self.grid

        cell_x = tf.reshape(tf.range(grid_width * grid_height,
                                     dtype='float32'), (grid_width, grid_height, 1, 1))
        cell_y = tf.transpose(cell_x, (1, 0, 2, 3))
        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [1, 1, 5, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        """
        Adjust prediction
        """
        # adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        # adjust w and h
        pred_box_wh = tf.exp(
            y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.max_grid_box, 2])

        # adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        # adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        # adjust x and y
        # relative position to the containing cell
        true_box_xy = y_true[..., 0:2]

        # adjust w and h
        # number of cells accross, horizontally and vertically
        true_box_wh = y_true[..., 2:4]

        # adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        # adjust class probabilities
        true_box_class = y_true[..., 5:]

        """
        Determine the masks
        """
        # coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

        # confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + \
            tf.cast(best_ious < 0.6, dtype='float32') * \
            (1 - y_true[..., 4]) * NO_OBJECT_SCALE

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

        # class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(
            self.class_weights, tf.argmax(true_box_class, -1)) * CLASS_SCALE

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(
            tf.cast(coord_mask > 0.0, dtype='float32'))
        nb_conf_box = tf.reduce_sum(tf.cast(conf_mask > 0.0, dtype='float32'))
        nb_class_box = tf.reduce_sum(
            tf.cast(class_mask > 0.0, dtype='float32'))

        loss_xy = tf.reduce_sum(
            tf.square(true_box_xy-pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(
            tf.square(true_box_wh-pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(
            tf.square(true_box_conf-pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        loss_class = categorical_crossentropy(
            y_true=true_box_class, y_pred=pred_box_class)
        loss_class = tf.reduce_sum(
            loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class
        return loss

    def summary(self):
        self.model.summary()

    def fit_generator(self, generator, batch_size=32, validation_data=None, epochs=10, verbose=0):

        checkpoint = ModelCheckpoint('weights.h5',
                                     monitor='val_loss',
                                     verbose=verbose,
                                     save_best_only=True,
                                     mode='min',
                                     save_freq=5)

        generator.set_batch_size(batch_size)

        for epoch in range(epochs):
            num_of_batches = generator.__len__()
            for batch_id in range(num_of_batches):
                batch_data = generator.__getitem__(batch_id)
                self.model.fit(
                    batch_data, batch_size=batch_size, verbose=verbose)
