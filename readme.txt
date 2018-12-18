数据和代码说明

./EMG_for_decomposition/S002下面的数据是由 Elements大硬盘/实验数据/肌电信号数据/simulated_sEMG/S002d/S00201d里面取出的
这个数据是合成数据，有混合的肌电信号数据和对应的MUAPTs的数据，可以用来做定量的评估。

./EMG_for_decomposition/swt 是我之前自己采集了自己的肌电信号数据，里面采集了两组实验trials。
./EMG_for_decomposition/prepare_data.py 就是用来处理这些数据的，把数据取出，按照512长度切出来。

.Elements/实验室/实验室数据/肌电信号数据/simulated_sEMG/S002d/S00201d/ 这个目录下面的代码是处理EMGlab的
里面的xml_reader.py 就是EMGlab提取出MUAP出现的时间点和波形之后，之间上这个就可以提取出来了。这个是非常重要的代码。
注意里面需要设置采样频率和采用时长！

本代码中用来训练的iEMG数据，就是利用EMGlab进行分解的数据，结合xml_reader.py进行运行的。

目录下，./results/plot_intervals.py 利用了自己编写的SpikeSorting的库，注意使用目录和方法。