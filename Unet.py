import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

from UnetUtils import UnetUtils
UnetUtils = UnetUtils()

class Unet():
    
    """ 
    Unet Model design.
    
    This module consumes the Unet utilities framework moule and designs the Unet network.
    It consists of a contracting path and an expansive path. Both these paths are joined 
    by a bottleneck block.
    
    The different blocks involved in the design of the network can be referenced @ 
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    
    Source:
        https://arxiv.org/pdf/1505.04597
    """
    
    def __init__(self):
        pass
    
    def Build_UNetwork(self, input_shape = (572, 572, 1)):
        
        """
        Builds the Unet Model network.
        
        Args:
            None
           
        Return:
            The Unet Model.
        
        """

        inputs = Input(input_shape)
        
        # the contracting path.
        conv1, pool1 = UnetUtils.contracting_block(input_layer = inputs, filters = 64)
        conv2, pool2 = UnetUtils.contracting_block(input_layer = pool1, filters = 128)
        conv3, pool3 = UnetUtils.contracting_block(input_layer = pool2, filters = 256)
        conv4, pool4 = UnetUtils.contracting_block(input_layer = pool3, filters = 512)
        
        # bottleneck block connecting the contracting 
        # and the expansive paths.
        bottleNeck = UnetUtils.bottleneck_block(pool4, filters = 1024)

        # the expansive path.
        upConv1 = UnetUtils.expansive_block(bottleNeck, conv4, filters = 512) 
        upConv2 = UnetUtils.expansive_block(upConv1, conv3, filters = 256) 
        upConv3 = UnetUtils.expansive_block(upConv2, conv2, filters = 128) 
        upConv4 = UnetUtils.expansive_block(upConv3, conv1, filters = 64) 

        outputs = Conv2D(1, (1, 1), padding = "valid", activation = "sigmoid")(upConv4)
        model = Model(inputs, outputs, name = "UNet")
        
        return model

    def CompileAndSummarizeModel(self, model):
        
        """
        Compiles and displays the model summary of the Unet model.
        
        Args:
            model: The Unet model.
        
        Return:
            None
        
        """
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        model.summary()
        
    def plotModel(self, model, to_file='unet.png', show_shapes=True, dpi=96):
        
        """
        Saves the Unet model to a file.
        
        Args:
            model: the Unet model. 
            to_file: the file name to save the model. Default name - 'unet.png'.
            show_shapes: whether to display shape information. Default = True.
            dpi: dots per inch. Default value is 96.
            
        Return:
            None
        
        """
        
        tf.keras.utils.plot_model(model, to_file = to_file, show_shapes = show_shapes, dpi = dpi)