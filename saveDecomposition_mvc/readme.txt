这个里面的.py较多，说明一下

show.py 是看运行decomposition_mvc_with_filter_optimizer.py的结果的，里面会显示找到的MUAPs，A，以及original EMG 和 approximated EMG的结果。

Cal_MUAPs_Kmeans.py 是按照z对MUAPTs进行分类的，其运行结束后会存储MUAPs_clusters.npy 里面是个字典，对应于MUAPTs的类别和波形，而存储的MUAPs_centers.npy里面存储的是距离KMeans中心最近的点的值

show_MUAPs_clusters.py 是看结果的，输出MUAPs_centers.npy里面存储的MUAPTs的spike time

show_MUAPs_according_to_A.py 是字面的意思，对于生成的MUAPTs，我们按照其A的值的大小进行排序，进而找到主要的MUAPs，里面还会对找到的主要的MUAPs计算其spike time。