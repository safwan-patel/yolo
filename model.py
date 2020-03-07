import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.layers import ReLU, LeakyReLU, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 32
WARM_UP_BATCHES  = 0
OBJECT_SCALE     = 5.0
NO_OBJECT_SCALE  = 1.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

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

    def init_layers(self):

        self.input_image = Input(shape=self.input_shape)
        self.true_boxes  = Input(shape=(1, 1, 1, self.max_image_box , 4))

        # Layer 1
        self.conv1 = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)
        self.norm1 = BatchNormalization(name='norm_1')
        self.activ1 = LeakyReLU(alpha=0.1)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        # Layer 2
        self.conv2 = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)
        self.norm2 = BatchNormalization(name='norm_2')
        self.activ2 = LeakyReLU(alpha=0.1)
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        # Layer 3
        self.conv3 = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)
        self.norm3 = BatchNormalization(name='norm_3')
        self.activ3 = LeakyReLU(alpha=0.1)

        # Layer 4
        self.conv4 = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)
        self.norm4 = BatchNormalization(name='norm_4')
        self.activ4 = LeakyReLU(alpha=0.1)

        # Layer 5
        self.conv5 = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)
        self.norm5 = BatchNormalization(name='norm_5')
        self.activ5 = LeakyReLU(alpha=0.1)
        self.pool5 = MaxPooling2D(pool_size=(2, 2))

        # Layer 6
        self.conv6 = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)
        self.norm6 = BatchNormalization(name='norm_6')
        self.activ6 = LeakyReLU(alpha=0.1)

        # Layer 7
        self.conv7 = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)
        self.norm7 = BatchNormalization(name='norm_7')
        self.activ7 = LeakyReLU(alpha=0.1)

        # Layer 8
        self.conv8 = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)
        self.norm8 = BatchNormalization(name='norm_8')
        self.activ8 = LeakyReLU(alpha=0.1)
        self.pool8 = MaxPooling2D(pool_size=(2, 2))

        # Layer 9
        self.conv9 = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)
        self.norm9 = BatchNormalization(name='norm_9')
        self.activ9 = LeakyReLU(alpha=0.1)

        # Layer 10
        self.conv10 = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)
        self.norm10 = BatchNormalization(name='norm_10')
        self.activ10 = LeakyReLU(alpha=0.1)

        # Layer 11
        self.conv11 = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)
        self.norm11 = BatchNormalization(name='norm_11')
        self.activ11 = LeakyReLU(alpha=0.1)

        # Layer 12
        self.conv12 = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)
        self.norm12 = BatchNormalization(name='norm_12')
        self.activ12 = LeakyReLU(alpha=0.1)

        # Layer 13
        self.conv13 = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)
        self.norm13 = BatchNormalization(name='norm_13')
        self.activ13 = LeakyReLU(alpha=0.1)
        self.pool13 = MaxPooling2D(pool_size=(2, 2))

        # Layer 14
        self.conv14 = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)
        self.norm14 = BatchNormalization(name='norm_14')
        self.activ14 = LeakyReLU(alpha=0.1)

        # Layer 15
        self.conv15 = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)
        self.norm15 = BatchNormalization(name='norm_15')
        self.activ15 = LeakyReLU(alpha=0.1)

        # Layer 16
        self.conv16 = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)
        self.norm16 = BatchNormalization(name='norm_16')
        self.activ16 = LeakyReLU(alpha=0.1)

        # Layer 17
        self.conv17 = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)
        self.norm17 = BatchNormalization(name='norm_17')
        self.activ17 = LeakyReLU(alpha=0.1)

        # Layer 18
        self.conv18 = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)
        self.norm18 = BatchNormalization(name='norm_18')
        self.activ18 = LeakyReLU(alpha=0.1)

        # Layer 19
        self.conv19 = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)
        self.norm19 = BatchNormalization(name='norm_19')
        self.activ19 = LeakyReLU(alpha=0.1)

        # Layer 20
        self.conv20 = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)
        self.norm20 = BatchNormalization(name='norm_20')
        self.activ20 = LeakyReLU(alpha=0.1)

        # Layer 21
        self.conv21 = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)
        self.norm21 = BatchNormalization(name='norm_21')
        self.activ21 = LeakyReLU(alpha=0.1)
        self.lamda21 = Lambda(lambda x: tf.nn.space_to_depth(x, block_size=2))

        # Layer 22
        self.conv22 = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)
        self.norm22 = BatchNormalization(name='norm_22')
        self.activ22 = LeakyReLU(alpha=0.1)

        # Layer 23
        self.conv23 = Conv2D(self.max_grid_box * (4 + 1 + len(self.labels)), (1,1), strides=(1,1), padding='same', name='conv_23')
        self.reshape23 = Reshape((self.grid[0], self.grid[1], self.max_grid_box, 4 + 1 + len(self.labels)))

    def build(self):
        
        # Layer 1
        x = self.conv1(self.input_image)
        x = self.norm1(x)
        x = self.activ1(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ2(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activ3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.activ4(x)

        # Layer 5
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.activ5(x)
        x = self.pool5(x)

        # Layer 6
        x = self.conv6(x)
        x = self.norm6(x)
        x = self.activ6(x)

        # Layer 7
        x = self.conv7(x)
        x = self.norm7(x)
        x = self.activ7(x)

        # Layer 8
        x = self.conv8(x)
        x = self.norm8(x)
        x = self.activ8(x)
        x = self.pool8(x)

        # Layer 9
        x = self.conv9(x)
        x = self.norm9(x)
        x = self.activ9(x)

        # Layer 10
        x = self.conv10(x)
        x = self.norm10(x)
        x = self.activ10(x)

        # Layer 11
        x = self.conv11(x)
        x = self.norm11(x)
        x = self.activ11(x)

        # Layer 12
        x = self.conv12(x)
        x = self.norm12(x)
        x = self.activ12(x)

        # Layer 13
        x = self.conv13(x)
        x = self.norm13(x)
        x = self.activ13(x)

        skip_connection = x

        x = self.pool13(x)

        # Layer 14
        x = self.conv14(x)
        x = self.norm14(x)
        x = self.activ14(x)

        # Layer 15
        x = self.conv15(x)
        x = self.norm15(x)
        x = self.activ15(x)

        # Layer 16
        x = self.conv16(x)
        x = self.norm16(x)
        x = self.activ16(x)

        # Layer 17
        x = self.conv17(x)
        x = self.norm17(x)
        x = self.activ17(x)

        # Layer 18
        x = self.conv18(x)
        x = self.norm18(x)
        x = self.activ18(x)

        # Layer 19
        x = self.conv19(x)
        x = self.norm19(x)
        x = self.activ19(x)

        # Layer 20
        x = self.conv20(x)
        x = self.norm20(x)
        x = self.activ20(x)

        # Layer 21
        skip_connection = self.conv21(skip_connection)
        skip_connection = self.norm21(skip_connection)
        skip_connection = self.activ21(skip_connection)
        skip_connection = self.lamda21(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = self.conv22(x)
        x = self.norm22(x)
        x = self.activ22(x)

        # Layer 23
        x = self.conv23(x)
        output = self.reshape23(x)
        self.model = Model([self.input_image, self.true_boxes], output, name=self.name)
    
    def compile(self, optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)):
        self.build()
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)
            
        
    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]
        grid_width, grid_height = self.grid
        
        cell_x = tf.reshape(tf.range(grid_width * grid_height, dtype='float32'), (1, grid_width, grid_height, 1, 1))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])
        
        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
        
        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        
        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1,1,1,self.max_grid_box,2])
        
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        
        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        true_box_conf = iou_scores * y_true[..., 4]
        
        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
        
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.cast(best_ious < 0.6, dtype='float32') * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
        
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE
        
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_weights, true_box_class) * CLASS_SCALE       
        
        """
        Warm-up training
        """
        no_boxes_mask = tf.cast(coord_mask < COORD_SCALE/2., dtype='float32')
        seen = seen.assign_add(1.)
        
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
                            lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                    true_box_wh + tf.ones_like(true_box_wh) * np.reshape(self.anchors, [1,1,1,self.max_grid_box,2]) * no_boxes_mask, 
                                    tf.ones_like(coord_mask)],
                            lambda: [true_box_xy, 
                                    true_box_wh,
                                    coord_mask])
        
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype='float32'))
        nb_conf_box  = tf.reduce_sum(tf.cast(conf_mask  > 0.0, dtype='float32'))
        nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, dtype='float32'))
        
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        
        loss = loss_xy + loss_wh + loss_conf + loss_class
        
        return loss

    def summary(self):
        self.model.summary()

    def fit_generator(self, generator, validation_data=None, epochs=10, verbose=0):
        early_stop = EarlyStopping(monitor='val_loss',
                                    min_delta=0.001, 
                                    patience=3, 
                                    mode='min', 
                                    verbose=verbose)

        checkpoint = ModelCheckpoint('weights.h5', 
                                    monitor='val_loss', 
                                    verbose=verbose, 
                                    save_best_only=True, 
                                    mode='min', 
                                    save_freq=5)

        self.model.fit_generator(generator = generator, 
                    steps_per_epoch  = len(generator),
                    epochs           = epochs, 
                    verbose          = verbose,
                    callbacks        = [early_stop, checkpoint], 
                    max_queue_size   = 3)
        

