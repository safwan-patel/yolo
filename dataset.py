import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from pycocotools.coco import COCO
from PIL import Image
from utils import bbox_iou

class Dataset:    
    def __init__(self, data_dir, data_instance='train2014'):
        annotation_file='{}/annotations/instances_{}.json'.format(data_dir, data_instance)
        self.dataset = COCO(annotation_file)
        self.image_dir = '%s/images/%s' % (data_dir,data_instance)

    def get_labels(self):
        catIds = self.dataset.getCatIds()
        cats = self.dataset.loadCats(catIds)
        return tuple(map(lambda cat: cat['name'], cats))

    def generator(self, image_shape=(416,416,3), grid=(13,13),
        anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
        max_grid_box=5, max_image_box=50, shuffle=True, jitter=True, norm=None):
        return Dataset.BatchGenerator(dataset = self.dataset,
                image_dir = self.image_dir,
                batch_size = batch_size,
                image_shape = image_shape,
                grid = grid,
                anchors = anchors,
                max_grid_box = max_grid_box,
                max_image_box = max_image_box,
                shuffle = shuffle,
                jitter = jitter,
                norm = norm)

    class BatchGenerator(Sequence):
        def __init__(self, dataset, image_dir, image_shape, grid,
            anchors, max_grid_box, max_image_box, shuffle, jitter, norm):
            self.dataset = dataset
            self.image_dir = image_dir
            self.image_width, self.image_height, self.image_channel = image_shape
            self.grid_width, self.grid_height = grid
            # anchors
            self.anchors = [[0, 0, anchors[2*i], anchors[2*i+1]] for i in range(int(len(anchors)//2))]
            self.max_grid_box = max_grid_box
            self.max_image_box = max_image_box
            self.jitter = jitter
            self.norm = norm

            self.img_ids = dataset.getImgIds()
            if shuffle: np.random.shuffle(self.img_ids)
            self.num_categories = len(self.dataset.getCatIds())
            # batch size
            self.batch_size = 32

        def set_batch_size(self, batch_size):
            self.batch_size = batch_size

        def __len__(self):
            'Number of batches per epoch'
            num_of_images = len(self.img_ids)
            return int(np.floor(num_of_images / self.batch_size))

        def __getitem__(self, idx):
            print('idx', idx)
            'Generate a batch of data'
            l_bound = idx * self.batch_size
            r_bound = (idx+1) * self.batch_size

            if r_bound > len(self.img_ids):
                r_bound = len(self.img_ids)
                l_bound = r_bound - self.batch_size
            
            imgs = self.dataset.loadImgs(self.img_ids[l_bound:r_bound])
            batch_item = np.array(tuple(map(self.get_item, imgs)))
            images = np.stack(batch_item[:,0], axis=0)
            true_boxes = np.stack(batch_item[:,2], axis=0)
            y_out = np.stack(batch_item[:,1], axis=0)
            return [images, y_out, true_boxes]

        def get_item(self, img):
            # get image object
            image, image_size = self.get_img(img['file_name'])
            
            # construct output from object's x, y, w, h
            true_box_index = 0
            
            # annotation
            y_anntn = np.zeros(shape=(self.grid_width, self.grid_height, self.max_grid_box, 4+1+self.num_categories))
            true_box = np.zeros(shape=(1, 1, 1,  self.max_image_box, 4))
            annIds = self.dataset.getAnnIds(imgIds=img['id'])
            annotations = self.dataset.loadAnns(annIds)

            for annotation in annotations:
                box = self.get_box(annotation, image_size=image_size)
                x, y, w, h = box
                
                grid_x = int(np.floor(x))
                grid_y = int(np.floor(y))
                
                # find the anchor that best predicts this box
                best_anchor = -1
                max_iou     = -1
                    
                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou([0, 0, w, h], anchor)
                    if max_iou < iou:
                        best_anchor = i
                        max_iou     = iou
                
                box_class = annotation['category_id']
                box_class = tf.one_hot(box_class, depth=self.num_categories, dtype='float32')
                y_anntn[grid_y, grid_x, best_anchor] = tf.concat([box, [1], box_class], axis=0)

                # assign the true box to b_batch
                true_box[0, 0, 0, true_box_index] = box

                true_box_index += 1
                true_box_index = true_box_index % self.max_image_box

            
            return image, y_anntn, true_box

        def get_img(self, file_name):
            image = Image.open('%s/%s' % (self.image_dir, file_name))
            size = image.size
            image = image.resize((self.image_width, self.image_height), Image.ANTIALIAS)
            img = np.asarray(image)
            image.close()
            return img, size

        def get_box(self, anntn, image_size):
            bbox = anntn['bbox']            
            xmin, ymin, b_width, b_height = bbox
            width, height = image_size
            center_x = (xmin + 0.5 * b_width) / float(width / self.grid_width)
            center_y = (ymin + 0.5 * b_height) / float(height / self.grid_height)
            center_w = b_width / float(width / self.grid_width)
            center_h = b_height / float(height / self.grid_height)
            return [center_x, center_y, center_w, center_h]
