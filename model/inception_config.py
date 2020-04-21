layer_3a = [[{
    "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (1, 1), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D", "filters" : 96, "kernel_size" : (1, 1), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
    "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (1, 1), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
    "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (5, 5), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
],[
    {
        "layer_type" : "MaxPooling2D",
        "strides" : 1,
        "kernel_size" : (3, 3),
        "padding" : "same"
    },{
    "layer_type" : "Conv2D",
    "filters" : 32,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
]]

layer_3b = [[{
    "layer_type" : "Conv2D",
    "filters" : 128,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 128,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 192,
    "kernel_size" : (3, 3),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 32,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 96,
    "kernel_size" : (5, 5),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
        "layer_type" : "MaxPooling2D",
        "strides" : 1,
        "kernel_size" : (3, 3),
        "padding" : "same"
    },{
    "layer_type" : "Conv2D",
    "filters" : 96,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
]]

layer_4a = [[{
    "layer_type" : "Conv2D",
    "filters" : 192,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 96,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 208,
    "kernel_size" : (3, 3),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 16,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 48,
    "kernel_size" : (5, 5),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
        "layer_type" : "MaxPooling2D",
        "strides" : 1,
        "kernel_size" : (3, 3),
        "padding" : "same"
    },{
    "layer_type" : "Conv2D",
    "filters" : 64,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
]]

layer_4b = [[{
    "layer_type" : "Conv2D",
    "filters" : 160,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 112,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 224,
    "kernel_size" : (3, 3),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 24,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 64,
    "kernel_size" : (5, 5),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
        "layer_type" : "MaxPooling2D",
        "strides" : 1,
        "kernel_size" : (3, 3),
        "padding" : "same"
    },{
    "layer_type" : "Conv2D",
    "filters" : 64,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
]]