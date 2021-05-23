import os
import sys

import numpy as np
import cv2

import tensorflow as tf

class NucleiDataGenerator(tf.keras.utils.Sequence):
    
    """
    The custom data generator class generates and feeds data to
    the model dynamically in batches during the training phase.
    
    This generator generates batched of data for the dataset available @
    Find the nuclei in divergent images to advance medical discovery -
    https://www.kaggle.com/c/data-science-bowl-2018
    
    **
    tf.keras.utils.Sequence is the root class for 
    Custom Data Generators.
    **
    
    Args:
        image_ids: the ids of the image.
        img_path: the full path of the image directory.
        batch_size: no. of images to be included in a batch feed. Default is set to 8.
        image_size: size of the image. Default is set to 128 as per the data available.
        
    Ref: https://dzlab.github.io/dltips/en/keras/data-generator/
    
    """
    def __init__(self, image_ids, img_path, batch_size = 8, image_size = 128):
        
        self.ids = image_ids
        self.path = img_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    def __load__(self, item):
        
        """
        loads the specified image.
        
        """
        
        # the name for parent of parent directory where the image is located and the name of the image are same.
        # an example directory breakup is shown below -
        # - data-science-bowl-2018/
        #      - stage1_train/
        #          - abc
        #             - image
        #                  - abc
        #             - mask
        full_image_path = os.path.join(self.path, item, "images", item) + ".png"
        mask_dir_path = os.path.join(self.path, item, "masks/")
        all_masks = os.listdir(mask_dir_path)
        
        # load the images
        image = cv2.imread(full_image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        masked_img = np.zeros((self.image_size, self.image_size, 1))
        
        # load and prepare the corresponding mask.
        for mask in all_masks:
            fullPath = mask_dir_path + mask
            _masked_img = cv2.imread(fullPath, -1)
            _masked_img = cv2.resize(_masked_img, (self.image_size, self.image_size))
            _masked_img = np.expand_dims(_masked_img, axis = -1)
            masked_img = np.maximum(masked_img, _masked_img)
            
        # mormalize the mask and the image. 
        image = image/255.0
        masked_img = masked_img/255.0
        
        return image, masked_img
    
    def __getitem__(self, index):
        
        """
        Returns a single batch of data.
        
        Args:
            index: the batch index.
        
        """
        
        # edge case scenario where there are still some items left
        # after segregatings the images into batches of size batch_size.
        # the items left out will form one batch at the end.
        if(index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size
        
        # group the items into a batch.
        batch = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        
        image = []
        mask  = []
        
        # load the items in the current batch
        for item in batch:
            img, masked_img = self.__load__(item)
            image.append(img)
            mask.append(masked_img)
        
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        
        """
        optional method to run some logic at the end of each epoch: e.g. reshuffling
        
        """
        
        pass
    
    def __len__(self):
        
        """
        Returns the number of batches
        """
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
    