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
    
    def Build_UNetwork(self, input_shape = (572, 572, 1), filters = [64, 128, 256, 512, 1024]):
        
        """
        Builds the Unet Model network.
        
        Args:
            input_shape: The shape of the input to the network. A tuple comprising of (img_height, img_width, channels).
                         Default shape is (572, 572, 1).
            filters: a collection of filters denoting the number of components to be used at each blocks along the 
                     contracting and expansive paths. The default number of filters along the contracting and expansive paths are
                     [64, 128, 256, 512, 1024].
         
        **Remarks: The default values are as per the implementation in the original paper @ https://arxiv.org/pdf/1505.04597
           
        Return:
            The Unet Model.
            
            **Note - If the total number of filters are not sufficient to implement each block along the contracting 
                     and expansive path, then the return value is None.
        
        """

        if len(filters) != 5:
            print("There are not sufficient filters to implement each block of the UNet model.\nRecheck the filters.")
            return None
        
        UnetInput = Input(input_shape)
        
        # the contracting path. 
        # the last item in the filetrs collection points to the number of filters in the bottleneck block.
        # so we loop till the 4th item.
        for num_filters in filters[:4]:
            conv1, pool1 = UnetUtils.contracting_block(input_layer = UnetInput, filters = num_filters)
            conv2, pool2 = UnetUtils.contracting_block(input_layer = pool1, filters = num_filters)
            conv3, pool3 = UnetUtils.contracting_block(input_layer = pool2, filters = num_filters)
            conv4, pool4 = UnetUtils.contracting_block(input_layer = pool3, filters = num_filters)
        
        # bottleneck block connecting the contracting and the expansive paths.
        bottleNeck = UnetUtils.bottleneck_block(pool4, filters = filters[-1])

        # the expansive path.essentially we loop the reversed filter list leaving out the last item.
        for num_filters in reversed(filters[:-1]):
            upConv1 = UnetUtils.expansive_block(bottleNeck, conv4, filters = num_filters) 
            upConv2 = UnetUtils.expansive_block(upConv1, conv3, filters = num_filters) 
            upConv3 = UnetUtils.expansive_block(upConv2, conv2, filters = num_filters) 
            upConv4 = UnetUtils.expansive_block(upConv3, conv1, filters = num_filters) 

        UnetOutput = Conv2D(1, (1, 1), padding = "valid", activation = "sigmoid")(upConv4)
        model = Model(UnetInput, UnetOutput, name = "UNet")
        
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