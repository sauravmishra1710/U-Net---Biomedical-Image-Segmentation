import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, Concatenate

class UnetUtils():
    
    """ 
    Unet Model design utillities framework.
    
    This module provides a convenient way to create different layers/blocks
    which the UNet network is based upon. It consists of a contracting
    path and an expansive path. Both these paths are joined by a bottleneck block.
    
    The different blocks involved in the design of the network can be referenced @ 
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    
    Source:
        https://arxiv.org/pdf/1505.04597
    """
    
    def __init__(self):
        pass
    
    def contracting_block(self, input_layer, filters, padding, kernel_size = 3):
        
        """ 
        UNet Contracting block
        Perform two unpadded convolutions with a specified number of filters and downsample
        through max-pooling.
        
        Args:
            input_layer: the input layer on which the current layers should work upon.
            filters (int): Number of filters in convolution.
            kernel_size (int/tuple): Index of block. Default is 3.
            padding ("valid" or "same"): Default is "valid" (no padding involved).
            
        Return:
            Tuple of convolved ``inputs`` after and before downsampling
        """
        
        # two 3x3 convolutions (unpadded convolutions), each followed by
        # a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
        # for downsampling.
        conv = Conv2D(filters = filters, 
                      kernel_size = kernel_size, 
                      activation = tf.nn.relu, 
                      padding = padding)(input_layer)

        conv = Conv2D(filters = filters, 
                      kernel_size = kernel_size, 
                      activation = tf.nn.relu, 
                      padding = padding)(conv)

        pool = MaxPooling2D(pool_size = 2, 
                            strides = 2)(conv)

        return conv, pool

    def bottleneck_block(self, input_layer, filters, padding, kernel_size = 3, strides = 1):
        
        """ 
        UNet bottleneck block
        
        Performs 2 unpadded convolutions with a specified number of filters.
        
        Args:
            input_layer: the input layer on which the current layers should work upon.
            filters (int): Number of filters in convolution.
            kernel_size (int/tuple): Index of block. Default is 3.
            padding ("valid" or "same"): Default is "valid" (no padding involved).
            strides: An integer or tuple/list of 2 integers, specifying the strides 
                     of the convolution along the height and width. Default is 1.
        Return:
            The convolved ``inputs``.
        """
        
        # two 3x3 convolutions (unpadded convolutions), each followed by
        # a rectified linear unit (ReLU)
        conv = Conv2D(filters = filters, 
                      kernel_size = kernel_size, 
                      padding = padding,
                      strides = strides, 
                      activation = tf.nn.relu)(input_layer)

        conv = Conv2D(filters = filters, 
                      kernel_size = kernel_size, 
                      padding = padding,
                      strides = strides, 
                      activation = tf.nn.relu)(conv)

        return conv

    def expansive_block(self, input_layer, skip_conn_layer, filters, padding, kernel_size = 3, strides = 1):
        
        """ 
        UNet expansive (upsample) block.
        
        Transpose convolution which doubles the spatial dimensions (height and width) 
        of the incoming feature maps and creates the skip connections with the corresponding 
        feature maps from the contracting (downsample) path. These skip connections bring the feature maps 
        from earlier layers helping the network to generate better semantic feature maps.
        
        Perform two unpadded convolutions with a specified number of filters 
        and upsamples the incomming feature map.
        
        Args:
            input_layer: the input layer on which the current layers should work upon.
            skip_connection: The feature map from the contracting (downsample) path from which the 
                             skip connection has to be created.
            filters (int): Number of filters in convolution.
            kernel_size (int/tuple): Index of block. Default is 3.
            padding ("valid" or "same"): Default is "valid" (no padding involved).
            strides: An integer or tuple/list of 2 integers, specifying the strides 
                     of the convolution along the height and width. Default is 1.
                     
        Return:
            The upsampled feature map.
        """
        
        # up sample the feature map using transpose convolution operations.
        transConv = Conv2DTranspose(filters = filters, 
                                    kernel_size = (2, 2),
                                    strides = 2, 
                                    padding = padding)(input_layer)
        
        # crop the source feature map so that the skip connection can be established.
        # the original paper implemented unpadded convolutions. So cropping is necessary 
        # due to the loss of border pixels in every convolution.
        # establish the skip connections.
        if padding == "valid":
            cropped = self.crop_tensor(skip_conn_layer, transConv)
            concat = Concatenate()([transConv, cropped])
        else:
            concat = Concatenate()([transConv, skip_conn_layer])
        
        # two 3x3 convolutions, each followed by a ReLU
        up_conv = Conv2D(filters = filters, 
                         kernel_size = kernel_size, 
                         padding = padding, 
                         activation = tf.nn.relu)(concat)

        up_conv = Conv2D(filters = filters, 
                         kernel_size = kernel_size, 
                         padding = padding, 
                         activation = tf.nn.relu)(up_conv)

        return up_conv
    
    def crop_tensor(self, source_tensor, target_tensor):
        
        """
        Center crops the source tensor to the size of the target tensor size.
        The tensor shape format is [batchsize, height, width, channels]
        
        Args:
            source_tensor: the tensor that is to be cropped.
            target_tensor: the tensor to whose size the 
                           source needs to be cropped to.
                           
        Return:
            the cropped version of the source tensor.
        
        """
        
        target_tensor_size = target_tensor.shape[2]
        source_tensor_size = source_tensor.shape[2]
        
        # calculate the delta to ensure correct cropping.
        delta = source_tensor_size - target_tensor_size
        delta = delta // 2
        
        cropped_source = source_tensor[:, delta:source_tensor_size - delta, delta:source_tensor_size - delta, :]
        
        return cropped_source