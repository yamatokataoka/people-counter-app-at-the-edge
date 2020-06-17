# Project Write-Up

## Explaining Custom Layers

The custom layer is defined as a layer, not in the [list of supported layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)

There are several options to convert custom layers regarding the original model framework you intend to use.

For example,

TensorFlow:
* register the custom layers as extensions to the Model Optimizer.
* replace the unsupported subgraph with a different subgraph.
* offload the computation of the subgraph back to TensorFlow during inference

Caffe:
* register the custom layers as extensions to the Model Optimizer.
* register the layers as Custom, then use Caffe to calculate the output shape of the layer.

The potential reason for handling custom layers is OpenVINO™ toolkit cannot optimize the model and run inference. The Inference Engine only considers the layer to be unsupported and reports an error.

## Comparing Model Performance

My method to compare models before and after conversion to Intermediate Representations
were running inference both on the OpenVINO toolkit and Tensorflow framework and measuring their performance (accuracy, size, speed, CPU overhead).

The difference between model accuracy pre- and post-conversion should be the same for accuracy unless you quantize it theoretically.

The size of the model pre- and post-conversion was squashed about 2 MB comparing .pb (Tensorflow) and .bin (IR).

The inference time of the model pre- and post-conversion was reduced by about 103 seconds, using [comparing_model_performance.py](./comparing_model_performance.py)

|            | size (MB) | inference time (s) |
|------------|-----------|--------------------|
| OpenVINO   | 96        | 222                |
| Tensorflow | 98        | 325                |

## Assess Model Use Cases

Some of the potential use cases of the people counter app is visitor analytics for the physical world using information like the number of visitors per store and time spent on certain places.

For example, in Airports, we can understand the traffic and track crowds movement from security check to plane embarkment.

These censoring allows more efficient operations, staff allocation, and increase passenger time at Duty-Free.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are model accuracy.

Changing environment parameters for an edge model like lack of lighting or camera performance leads to decrease model accuracy because input image could be far away from trained images in the first place. Like so, the insufficient input image size will decrease the model accuracy.
