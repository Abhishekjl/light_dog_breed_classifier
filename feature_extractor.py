from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as resnet_preprocess
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
from keras.applications.xception import Xception, preprocess_input as xception_preprocess
from keras.applications.nasnet import NASNetLarge, preprocess_input as nasnet_preprocess
from keras.layers.merge import concatenate
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

def extractor_model():
    input_shape = (331,331,3)
    input_layer = Input(shape=input_shape)


    #first extractor inception_resnet
    preprocessor_resnet = Lambda(resnet_preprocess)(input_layer)
    inception_resnet = InceptionResNetV2(weights = 'imagenet',
                                     include_top = False,input_shape = input_shape,pooling ='avg')(preprocessor_resnet)

# second extractor InceptionV3
    preprocessor_inception = Lambda(inception_preprocess)(input_layer)
    inception_v3 = InceptionV3(weights = 'imagenet',
                                     include_top = False,input_shape = input_shape,pooling ='avg')(preprocessor_inception)
# Third extractor Xception
    preprocessor_xception = Lambda(xception_preprocess)(input_layer)
    xception = Xception(weights = 'imagenet',
                                     include_top = False,input_shape = input_shape,pooling ='avg')(preprocessor_xception)

# fourth extractor nasnet large
    preprocessor_nasnet = Lambda(nasnet_preprocess)(input_layer)
    nasnet = NASNetLarge(weights = 'imagenet',
                                     include_top = False,input_shape = input_shape,pooling ='avg')(preprocessor_nasnet)


    merge = concatenate([inception_v3, xception,nasnet,inception_resnet])
    model = Model(inputs = input_layer, output = merge)
    return model