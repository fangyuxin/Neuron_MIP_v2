# 神经元突触重建项目——Neuron_MIP_v2

使用框架: PyTorch

网络: Unet


全部结果在[all results](https://github.com/fangyuxin/Neuron_MIP_v2/blob/master/all_results.zip)压缩包内

部分结果在[some results](https://github.com/fangyuxin/Neuron_MIP_v2/tree/master/some_results)文件夹内，其中[result_stat.csv](https://github.com/fangyuxin/Neuron_MIP_v2/tree/master/some_results)记录了结果的评价指标。

下面是一些例子。其中最左侧的是成像设备得到的原始图(input)，中间是标签(label)，最右侧是模型的输出(output)。

从下图不难看出，对比输入，模型的output输出了本应存在但是label中没有的神经元连接(第40号图):
![40](https://github.com/fangyuxin/Neuron_MIP_v2/blob/master/some_results/result_img_(0%7E90)/result_40.png)

当然，也有不足之处，例如对于连接比较丰富的情况，输出的连接关系不够理想(第33号图):
![40](https://github.com/fangyuxin/Neuron_MIP_v2/blob/master/some_results/result_img_(0%7E90)/result_33.png)
