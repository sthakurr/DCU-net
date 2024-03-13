
def set_pooling_stride(model):
    for layer in model.layers:
        if isinstance(layer, MaxPooling2D):
            layer.strides = (2,2)
    return model
