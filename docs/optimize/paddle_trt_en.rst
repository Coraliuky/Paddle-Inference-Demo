Using the Paddle-TensorRT Repository for Inference
================

NVIDIA TensorRT is a high-performance inference repository of deep learning, and it can lower the latency of the inference applications of deep learning and improve their throughput. PaddlePaddle uses subgraph of integrate TensorRT, so we can use the module to enhance the performance of the Paddle model in inference. In this article, we will talk about how to use the subgraph module of Paddle-TRT to accelerate the inference. 

If you need to install `TensorRT <https://developer.nvidia.com/nvidia-tensorrt-6x-download>`_, please refer to the `trt document <https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html>`_.

Overview
----------------

After the model is loaded, the neural network can be represented as a computing chart consisting of variables and computing nodes. If the TRT subgraph mode is turned on, Paddle will analyze the model graph, find out subgraphs which can be optimized by TensorRT there in the analysis stage, and replace them with TensorRT nodes. During the model inference, if encountering TensorRT nodes, Paddle will optimize the modes with the TensorRT repository，and optimize other nodes with the original implementation of Paddle. Besides the common OP integration and the video memory/ memory optimization, TensorRT also acclerates the implementation of OP, lowers the inference latency, and improve the throughput. 

Paddle-TRT support the static shape mode and the dynamic shape mode currently. In the static mode, image classification and segmentation and model detection are available. Also, the inference acceleration of Fp16 and Int8 are supported. In the dynamic mode, in addition to the image models (FCN, Faster rcnn) of the dynamic shape, Bert/Ernie of NLP are also supported.

**Capabilities of Paddle-TRT：**

**1）Static shape：**

Supported models：

===============  ===============  =============
 Classification    Detection       Segmentation  
 Models            Models          Models
===============  ===============  =============
Mobilenetv1        yolov3             ICNET
Resnet50           SSD                UNet
Vgg16              Mask-rcnn          FCN
Resnext            Faster-rcnn
AlexNet            Cascade-rcnn
Se-ResNext         Retinanet
GoogLeNet          Mobilenet-SSD
DPN
===============  ===============  =============

.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |check_| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

.. |uncheck_| raw:: html

    <input disabled="" type="checkbox">

Fp16: |check|

Calib Int8: |check|

Serialize optimized information: |check|

Load the PaddleSlim Int8 model: |check|


**2）Dynamic shape：**

Supported models：

===========  =====
   Images     NLP
===========  =====
FCN          Bert
Faster_RCNN  Ernie
===========  =====

Fp16: |check|

Calib Int8: |uncheck|

Serialize optimized information: |uncheck|

Load the PaddleSlim Int8 model: |uncheck|


**Note:**

1. During the compilation of the source code, the TensorRT inference repository only supports GPU compilation, and TENSORRT_ROOT is required to be set to the path of TensorRT. 
2. Only TensorRT versions above 5.0 are supported by Windows.
3. The version of TRT  should be above 6.0 if the input of the dynamic shape uses Paddle-TRT.


I. Environment Preparation
-------------

To use the functions of Paddle-TRT, the runtime environment of Paddle containing TRT is required. There are three ways to get prepared: 

1）Using pip to install a whl file under linux

Download a whl file with the consistent environment and trt from `whl list <https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release>`_ , and install it using pip. 

2）Using the docker

.. code:: shell

	# Pull the docker, which is already preinstalled the Paddle 2.2 Python environment and contains a precompiled library (c++) put in the main directory ～/.
	docker pull paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82

	sudo nvidia-docker run --name your_name -v $PWD:/paddle  --network=host -it paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82  /bin/bash

3）Manual Compilation  
Please refer to the `compilation document <../user_guides/source_compile.html>`_ 

**Note1：** During the cmake, please set TENSORRT_ROOT （the path of TRT lib）and WITH_PYTHON （set "whether to produce the python whl file" to ON).

**Note2:** There will be errors of TensorRT during the compilation.

Add virtual destructors to class IPluginFactory and class IGpuAllocator of NvInfer.h (trt5) or NvInferRuntime.h (trt6) file respectively by hand:

.. code:: c++

	virtual ~IPluginFactory() {};
	virtual ~IGpuAllocator() {};
	
Change **protected: ~IOptimizationProfile() noexcept = default;** in `NvInferRuntime.h` (trt6)

to

.. code:: c++

	virtual ~IOptimizationProfile() noexcept = default;
	


II. Introduction to the usage of APIs
-----------------

In the section of `the inference process <https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html>`_ , we have got to know that there are five parts of Paddle Inference:

- Configuration of inference options
- Creation of the predictor
- Preparation for the model input
- Model inference
- Acquisition of the model ouput

Paddle-TRT also follows the same process. Let's use a simple example to introduce it (It is assumed that you have known about the Paddle Inference). If you are new to this, you can visit <https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html>`_ to get started.

.. code:: python

    import numpy as np
    import paddle.inference as paddle_infer
    
    def create_predictor():
        config = paddle_infer.Config("./resnet50/model", "./resnet50/params")
        config.enable_memory_optim()
        config.enable_use_gpu(1000, 0)
        
        # Open TensorRT. The details of this interface will be mentioned in the following part.
        config.enable_tensorrt_engine(workspace_size = 1 << 30, 
                                      max_batch_size = 1, 
                                      min_subgraph_size = 3, 
                                      precision_mode=paddle_infer.PrecisionType.Float32, 
                                      use_static = False, use_calib_mode = False)

        predictor = paddle_infer.create_predictor(config)
        return predictor

    def run(predictor, img):
        # Preparation for the input
        input_names = predictor.get_input_names()
        for i,  name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(img[i].shape)   
            input_tensor.copy_from_cpu(img[i].copy())
        # Inference
        predictor.run()
        results = []
        # Acquisition of the output
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        return results

    if __name__ == '__main__':
        pred = create_predictor()
        img = np.ones((1, 3, 224, 224)).astype(np.float32)
        result = run(pred, [img])
        print ("class index: ", np.argmax(result[0][0]))


From this example, it is clear that we open TensorRT options through the interface of `enable_tensorrt_engine`.

.. code:: python

    config.enable_tensorrt_engine(workspace_size = 1 << 30, 
                                  max_batch_size = 1, 
                                  min_subgraph_size = 3, 
                                  precision_mode=paddle_infer.PrecisionType.Float32, 
                                  use_static = False, use_calib_mode = False)

Then, let's have a look at the function of each parameter in the interface:

- **workspace_size**，type：int，and the default value is 1 << 30 （1G）. It designates the size of the working space of TensorRT, and TensorRT will sort out the optimum kernel for the execution of the inference computation under this limitation. 
- **max_batch_size**，type：int，and the default value is 1. The maximum batch is required to be set beforehand, and the batch size cannot exceed this max value in the execution. 
- **min_subgraph_size**，type：int，and the default value is 3. Paddle-TRT is operated in subgraphs. In order to avoid performance loss, Paddle-TRT will be operated only when the number of nodes within subgraphs is more than min_subgraph_size.
- **precision_mode**，type: **paddle_infer.PrecisionType**, and the default value is **paddle_infer.PrecisionType.Float32**. It designates the precision of TRT, and supports FP32（Float32）,FP16（Half）,and Int8（Int8）. If you need to use the post-training quantization calibration of Paddle-TRT int8, set the precision to **paddle_infer.PrecisionType.Int8** and **use_calib_mode** to True.
- **use_static**，type：bool, and the default value is False. If it is designated as True, then the optimized TRT information will be serialized to the disk during the first run of the program, and will be directly loaded next time without regeneration.
- **use_calib_mode**，type：bool, and the default value is False. If you need to use the post-training quantization calibration of Paddle-TRT int8, set this to True. 

Int8 Quantization Forecast
>>>>>>>>>>>>>>

神经网络的参数在一定程度上是冗余的，在很多任务上，我们可以在保证模型精度的前提下，将Float32的模型转换成Int8的模型，从而达到减小计算量降低运算耗时、降低计算内存、降低模型大小的目的。使用Int8量化预测的流程可以分为两步：1）产出量化模型；2）加载量化模型进行Int8预测。下面我们对使用Paddle-TRT进行Int8量化预测的完整流程进行详细介绍。

**1. 产出量化模型**

目前，我们支持通过两种方式产出量化模型：

a. 使用TensorRT自带Int8离线量化校准功能。校准即基于训练好的FP32模型和少量校准数据（如500～1000张图片）生成校准表（Calibration table），预测时，加载FP32模型和此校准表即可使用Int8精度预测。生成校准表的方法如下：

  - 指定TensorRT配置时，将 **precision_mode** 设置为 **paddle_infer.PrecisionType.Int8** 并且设置 **use_calib_mode** 为 **True**。

    .. code:: python

      config.enable_tensorrt_engine(
        workspace_size=1<<30,
        max_batch_size=1, min_subgraph_size=5,
        precision_mode=paddle_infer.PrecisionType.Int8,
        use_static=False, use_calib_mode=True)

  - 准备500张左右的真实输入数据，在上述配置下，运行模型。（Paddle-TRT会统计模型中每个tensor值的范围信息，并将其记录到校准表中，运行结束后，会将校准表写入模型目录下的 `_opt_cache` 目录中）

  如果想要了解使用TensorRT自带Int8离线量化校准功能生成校准表的完整代码，请参考 `这里 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt/README.md#%E7%94%9F%E6%88%90%E9%87%8F%E5%8C%96%E6%A0%A1%E5%87%86%E8%A1%A8>`_ 的demo。

b. 使用模型压缩工具库PaddleSlim产出量化模型。PaddleSlim支持离线量化和在线量化功能，其中，离线量化与TensorRT离线量化校准原理相似；在线量化又称量化训练(Quantization Aware Training, QAT)，是基于较多数据（如>=5000张图片）对预训练模型进行重新训练，使用模拟量化的思想，在训练阶段更新权重，实现减小量化误差的方法。使用PaddleSlim产出量化模型可以参考文档：
  
  - 离线量化 `快速开始教程 <https://paddlepaddle.github.io/PaddleSlim/quick_start/quant_post_tutorial.html>`_
  - 离线量化 `API接口说明 <https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-post>`_
  - 离线量化 `Demo <https://github.com/PaddlePaddle/PaddleSlim/tree/release/1.1.0/demo/quant/quant_post>`_
  - 量化训练 `快速开始教程 <https://paddlepaddle.github.io/PaddleSlim/quick_start/quant_aware_tutorial.html>`_
  - 量化训练 `API接口说明 <https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-aware>`_
  - 量化训练 `Demo <https://github.com/PaddlePaddle/PaddleSlim/tree/release/1.1.0/demo/quant/quant_aware>`_

离线量化的优点是无需重新训练，简单易用，但量化后精度可能受影响；量化训练的优点是模型精度受量化影响较小，但需要重新训练模型，使用门槛稍高。在实际使用中，我们推荐先使用TRT离线量化校准功能生成量化模型，若精度不能满足需求，再使用PaddleSlim产出量化模型。
  
**2. 加载量化模型进行Int8预测**       

  加载量化模型进行Int8预测，需要在指定TensorRT配置时，将 **precision_mode** 设置为 **paddle_infer.PrecisionType.Int8** 。

  若使用的量化模型为TRT离线量化校准产出的，需要将 **use_calib_mode** 设为 **True** ：

  .. code:: python

    config.enable_tensorrt_engine(
      workspace_size=1<<30,
      max_batch_size=1, min_subgraph_size=5,
      precision_mode=paddle_infer.PrecisionType.Int8,
      use_static=False, use_calib_mode=True)

  完整demo请参考 `这里 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt/README.md#%E5%8A%A0%E8%BD%BD%E6%A0%A1%E5%87%86%E8%A1%A8%E6%89%A7%E8%A1%8Cint8%E9%A2%84%E6%B5%8B>`_ 。
  
  若使用的量化模型为PaddleSlim量化产出的，需要将 **use_calib_mode** 设为 **False** ：

  .. code:: python

    config.enable_tensorrt_engine(
      workspace_size=1<<30,
      max_batch_size=1, min_subgraph_size=5,
      precision_mode=paddle_infer.PrecisionType.Int8,
      use_static=False, use_calib_mode=False)

  完整demo请参考 `这里 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt/README.md#%E4%B8%89%E4%BD%BF%E7%94%A8trt-%E5%8A%A0%E8%BD%BDpaddleslim-int8%E9%87%8F%E5%8C%96%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B>`_ 。

运行Dynamic shape
>>>>>>>>>>>>>>

从1.8 版本开始， Paddle对TRT子图进行了Dynamic shape的支持。
使用接口如下：

.. code:: python

	config.enable_tensorrt_engine(
		workspace_size = 1<<30,
		max_batch_size=1, min_subgraph_size=5,
		precision_mode=paddle_infer.PrecisionType.Float32,
		use_static=False, use_calib_mode=False)
		  
	min_input_shape = {"image":[1,3, 10, 10]}
	max_input_shape = {"image":[1,3, 224, 224]}
	opt_input_shape = {"image":[1,3, 100, 100]}

	config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape, opt_input_shape)



从上述使用方式来看，在 config.enable_tensorrt_engine 接口的基础上，新加了一个config.set_trt_dynamic_shape_info 的接口。     

该接口用来设置模型输入的最小，最大，以及最优的输入shape。 其中，最优的shape处于最小最大shape之间，在预测初始化期间，会根据opt shape对op选择最优的kernel。   

调用了 **config.set_trt_dynamic_shape_info** 接口，预测器会运行TRT子图的动态输入模式，运行期间可以接受最小，最大shape间的任意的shape的输入数据。



三：测试样例
-------------

我们在github上提供了使用TRT子图预测的更多样例：

- Python 样例请访问此处 `链接 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python/paddle_trt>`_ 。
- C++ 样例地址请访问此处 `链接 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt>`_ 。

四：Paddle-TRT子图运行原理
---------------

   PaddlePaddle采用子图的形式对TensorRT进行集成，当模型加载后，神经网络可以表示为由变量和运算节点组成的计算图。Paddle TensorRT实现的功能是对整个图进行扫描，发现图中可以使用TensorRT优化的子图，并使用TensorRT节点替换它们。在模型的推断期间，如果遇到TensorRT节点，Paddle会调用TensorRT库对该节点进行优化，其他的节点调用Paddle的原生实现。TensorRT在推断期间能够进行Op的横向和纵向融合，过滤掉冗余的Op，并对特定平台下的特定的Op选择合适的kernel等进行优化，能够加快模型的预测速度。  

下图使用一个简单的模型展示了这个过程：  

**原始网络**

	.. image:: https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_original.png

**转换的网络**

	.. image:: https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_trt.png

 我们可以在原始模型网络中看到，绿色节点表示可以被TensorRT支持的节点，红色节点表示网络中的变量，黄色表示Paddle只能被Paddle原生实现执行的节点。那些在原始网络中的绿色节点被提取出来汇集成子图，并由一个TensorRT节点代替，成为转换后网络中的 **block-25** 节点。在网络运行过程中，如果遇到该节点，Paddle将调用TensorRT库来对其执行。
