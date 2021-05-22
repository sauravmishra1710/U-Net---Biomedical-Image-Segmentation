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
    
    def __init__(self, input_shape = (572, 572, 1), filters = [64, 128, 256, 512, 1024], padding = "valid"):
        """
        
        Initialize the Unet framework and the model parameters - input_shape, 
        filters and padding type. 
        
        Args:
            input_shape: The shape of the input to the network. A tuple comprising of (img_height, img_width, channels).
                         Original paper implementation is (572, 572, 1).
            filters: a collection of filters denoting the number of components to be used at each blocks along the 
                     contracting and expansive paths. The original paper implementation for number of filters along the 
                     contracting and expansive paths are [64, 128, 256, 512, 1024].
            padding: the padding type to be used during the convolution step. The original paper used unpadded convolutions 
                     which is of type "valid".
         
        **Remarks: The default values are as per the implementation in the original paper @ https://arxiv.org/pdf/1505.04597
        
        """
        self.input_shape = input_shape
        self.filters = filters
        self.padding = padding
    
    def Build_UNetwork(self):
        
        """
        Builds the Unet Model network.
        
        Args:
            None
         
        Return:
            The Unet Model.
            
        """

        
        UnetInput = Input(self.input_shape)
        
        # the contracting path. 
        # the last item in the filetrs collection points to the number of filters in the bottleneck block.
        conv1, pool1 = UnetUtils.contracting_block(input_layer = UnetInput, filters = self.filters[0], padding = self.padding)
        conv2, pool2 = UnetUtils.contracting_block(input_layer = pool1, filters = self.filters[1], padding = self.padding)
        conv3, pool3 = UnetUtils.contracting_block(input_layer = pool2, filters = self.filters[2], padding = self.padding)
        conv4, pool4 = UnetUtils.contracting_block(input_layer = pool3, filters = self.filters[3], padding = self.padding)
        
        # bottleneck block connecting the contracting and the expansive paths.
        bottleNeck = UnetUtils.bottleneck_block(pool4, filters = self.filters[4], padding = self.padding)

        # the expansive path.
        upConv1 = UnetUtils.expansive_block(bottleNeck, conv4, filters = self.filters[3], padding = self.padding) 
        upConv2 = UnetUtils.expansive_block(upConv1, conv3, filters = self.filters[2], padding = self.padding) 
        upConv3 = UnetUtils.expansive_block(upConv2, conv2, filters = self.filters[1], padding = self.padding) 
        upConv4 = UnetUtils.expansive_block(upConv3, conv1, filters = self.filters[0], padding = self.padding) 

        UnetOutput = Conv2D(1, (1, 1), padding = self.padding, activation = tf.math.sigmoid)(upConv4)
        
        model = Model(UnetInput, UnetOutput, name = "UNet")
        
        return model

    def CompileAndSummarizeModel(self, model, optimizer = "adam", loss = "binary_crossentropy"):
        
        """
        Compiles and displays the model summary of the Unet model.
        
        Args:
            model: The Unet model.
            optimizer: model optimizer. Default is the adam optimizer.
            loss: the loss function. Default is the binary cross entropy loss.
            
        Return:
            None
        
        """
        model.compile(optimizer = optimizer, loss = loss, metrics = ["acc"])
        model.summary()
        
    def plotModel(self, model, to_file = 'unet.png', show_shapes = True, dpi = 96):
        
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