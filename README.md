# Xiao Song's HPC & Parallel Computation Note

## Ackmowlegement
This repo contains my note on self-learning HPC & Parallel Computation. I am releasing my note because I wish my note could help people who are interested in HPC & Parallel Computing and want to learn it themselves. 

One should notice that I do not contain the copyright of some image files. The copyright belongs to the original author. Any content inside this repo is NOT ALLOWED FOR COMMERCIAL USE.

I have tried to include all the reference material I used for the corresponding content. This is intended to encourage the user to read those raw reference material themselves as my note only include my undestanding.

Correction on mistakes is highly welcomed. Please post a issue if you found one.

To open markdown file with better format, [typora](https://typora.io) is recommended.

Many thanks to professor James at Berkeley for the excellent CS267 class. Many thanks for my mentor at Yitu Tech for they introduce me to the field of HPC.




这份报告包含了我关于自学习HPC和并行计算的笔记。我发布我的笔记是因为我希望我的笔记能帮助那些对HPC和并行计算感兴趣并想自己学习的人。

人们应该注意到，我不拥有一些图像文件的版权。版权归原作者所有。本文件中的任何内容均不得用于商业用途。

我尝试批注所有的参考资料。这是为了鼓励用户自己阅读这些原始参考资料，因为我的注释只包括我的理解。

我非常欢迎纠正错误。如果发现问题，请使用issue发布问题。

要以更好的格式打开.md文件，[typora](https://typora.io)推荐使用。

非常感谢Berkeley的James教授为我们提供了出色的CS267课程。非常感谢我在依图科技的导师，他们向我介绍了HPC领域。



## File Structure

```shell
Resources.md   # Useful resources for learning HPC
Basic.md # Contain basic knowlege
OptimizeSerialCode.md # How to optimize serial code
GPU-CUDA.md # CUDA related knowlege
Compiler.md # Compiler related optimization
ThreeParallelModel # 3 parallel model of programming
Scheduling.md # scheduling related topic
CPUMemoryCache.md # CPU Memory Cache related topic
ToolsLibrary.md # tools and library related
CommonAlgorithm.md # contain common algorithm related
DenseLinearAlgebra-CommunicationLowerBound.md # communication optimal algorithm (n-body, gemm, conv) & Dense Linear Algebra related (including GEMM)
ParallelMLTraining.md # parallel machine learning training (data parallel, model paralle, operator parallel, stc)
Numerical.md # numerical stability, representation related topics
```



## Major References

* Courses
  * UIUC ECE 408
  * UIUC ECE 508
  * UC Berkeley CS267
  * CMU 15-618
  * MIT 6.172
* Book
  * Programming Massively Parallel Processors 3rd
  * CUDA C++ Best Practices Guide
  * CUDA C++ Programming Guide
* Papers
  * Anatomy of High-Performance Matrix Multiplication
  * Roofline An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures

