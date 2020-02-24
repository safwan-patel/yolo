import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from pycocotools.coco import COCO
from PIL import Image

class Dataset:    
    def __init__(self, data_dir, data_instance='train2014'):
        annotation_file='{}/annotations/instances_{}.json'.format(data_dir, data_instance)
        self.dataset = COCO(annotation_file)
        self.image_dir = '%s/images/%s' % (data_dir,data_instance)

    def get_labels(self):
        catIds = self.dataset.getCatIds()
        cats = self.dataset.loadCats(catIds)
        return tuple(map(lambda cat: cat['name'], cats))

    def generator(self, batch_size=32, box=32, shuffle=True, jitter=True, norm=None):
        return Dataset.BatchGenerator(dataset = self.dataset,
                image_dir = self.image_dir,
                batch_size = batch_size,
                box = box,
                shuffle = shuffle,
                jitter = jitter,
                norm = norm)

    class BatchGenerator(Sequence):
        def __init__(self, dataset, image_dir, batch_size, box, shuffle, jitter, norm):
            self.dataset = dataset
            self.image_dir = image_dir
            self.batch_size = batch_size
            self.box = box
            self.jitter = jitter
            self.norm = norm
            self.img_ids = dataset.getImgIds()
            if shuffle: np.random.shuffle(self.img_ids)
            self.num_categories = len(self.dataset.getCatIds())

        def __len__(self):
            'Number of batches per epoch'
            num_of_images = len(self.img_ids)
            return int(np.floor(num_of_images / self.batch_size))

        def __getitem__(self, idx):
            'Generate a batch of data'
            img_ids = self.img_ids[idx * self.batch_size: (idx+1) * self.batch_size]
            imgs = self.dataset.loadImgs(img_ids)
            batch_item = np.array(tuple(map(self.get_item, imgs)))
            return batch_item[:,0], batch_item[:,1]

        def get_item(self, img):
            annIds = self.dataset.getAnnIds(imgIds=img['id'])
            annotations = self.dataset.loadAnns(annIds)
            bboxes = tf.convert_to_tensor(tuple(map(self.get_annotation, annotations)))
            img = self.get_img(img)
            return img, bboxes

        def get_img(self, img):
            I = Image.open('%s/%s' % (self.image_dir, img['file_name']))
            return np.asarray(I)

        def get_annotation(self, anntn):
            bbox = tf.convert_to_tensor(anntn['bbox'], dtype='float32')
            box_class = anntn['category_id']
            box_class = tf.one_hot(box_class, depth=self.num_categories, dtype='float32')
            y = tf.concat([bbox, [1], box_class], axis=0)
            return y
