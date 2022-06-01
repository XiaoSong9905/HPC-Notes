# Numerical

> Reference
>
> 1. PMPP Chapter 6



## Data representation

#### IEEE Standard

IEEE-753 Floating Point Standard 是硬件与软件使用的标准

`S` sign bit

`E` exponent bit

`M` mantissa bit



single precision float : 1 bit S, 8 bit E, 23 bit M

double precision float : 1 bit S, 11 bit E, 52 bit M



### Normalized representation



<img src="Note.assets/Screen Shot 2022-06-01 at 2.21.54 PM.png" alt="Screen Shot 2022-06-01 at 2.21.54 PM" style="zoom:50%;" />



下面的图片展示了normalized float point的representable number

<img src="Note.assets/Screen Shot 2022-06-01 at 2.25.08 PM.png" alt="Screen Shot 2022-06-01 at 2.25.08 PM" style="zoom:50%;" />



#### M Bit

在normalized representation下，要求M的形式是 `1.M`, 这样保证每一个float都有unique mantissa bit

number of mantissa bit 用于 floating point **precision**

有n个m bit，每个major interval就会有2^n个representable number。如果value无法被representable number表示，就会被rounded。

因为normalizd representation的限制，靠近0的部部分(0-0.5) 有representation gap，这个部分的数据无法被表达，要不被round到0，要不被round到0.5

`0` 无法用上面的normalized representation公式表达，这是一个很大的问题。

M bit每多一位，major interval中可以表达的number就多一倍，accuracy就会多一倍



#### E Bit

使用 2's complement `excess represerntation`, 这样可以通过比较bit来直接比较floating point的大小，让hw上实现起来更快

<img src="Note.assets/Screen Shot 2022-06-01 at 2.24.29 PM.png" alt="Screen Shot 2022-06-01 at 2.24.29 PM" style="zoom:50%;" />

number of exponent bit 用于 floating point **range**

有n个e bit，就会有n+1个interval。Figure 6.5中有三个major interval，分别对应

<img src="Note.assets/Screen Shot 2022-06-01 at 2.42.41 PM.png" alt="Screen Shot 2022-06-01 at 2.42.41 PM" style="zoom:50%;" />

major interval的大小距离0越近，interval大小越近。因为每个interval有固定数量个representable number，所以靠近0的interval的precision越大。对于很多division和converge的问题，这是大的问题



### Denormalized representation

当E=0的时候，mantissa的constrain没有，assume `0.M` 的形式而不是 `1.M` 的形式。

从理解的角度上来说，就是把靠近0的major interval spread out在0到interval end之间。原来靠近0的interval是(0.5, 1)有4个representable number（4是因为假设的Mbit个数），现在把这4个representable number分散在(0, 1)之间

<img src="Note.assets/Screen Shot 2022-06-01 at 2.47.17 PM.png" alt="Screen Shot 2022-06-01 at 2.47.17 PM" style="zoom:50%;" />





#### GPU Hardware

支持denormalized representaiton的硬件比较复杂，因为需要判断e是否是0，从而决定对应的M的format (1.M, 0.M)

computaiton capacity 1.3+ 支持 denormalized double floating point

computation capacity 2.0+ 支持 denormalized single floating point



#### Special Bit

<img src="Note.assets/Screen Shot 2022-06-01 at 2.52.02 PM.png" alt="Screen Shot 2022-06-01 at 2.52.02 PM" style="zoom:50%;" />



## Arithmetic accuracy

### Accuracy from Hardware

计算中的accuracy问题是因为计算的结果（或者临时结果）无法被准确表达，被rounding，所以导致accuracy问题。

plus minus，硬件accuracy控制在 0.5D ULP (units in last place)

division transcendental通过polynomial approximate实现，硬件accuracy一般比加减的accuracy要大



### Accuracy from Algorithm / Software

#### 大数吃小数

##### 原因

越靠近0（小数）的precision越大，越靠近inf（大数）的precision越小。

大数+小数=大数，结果的大数会因为rounding的问题无法表达被加上来的小数。



例子：sequential相加，从大数加到小数，结果中小数被忽略

<img src="Note.assets/Screen Shot 2022-06-01 at 2.56.28 PM.png" alt="Screen Shot 2022-06-01 at 2.56.28 PM" style="zoom:50%;" />



##### 解决方法

sort input data，然后sequential相加，这样很多小数彼此相加就能得到足够大的数，再与大数相加的时候不会被吃掉

<img src="Note.assets/Screen Shot 2022-06-01 at 2.56.58 PM.png" alt="Screen Shot 2022-06-01 at 2.56.58 PM" style="zoom:50%;" />



#### Numerical stable / unstable

If an algorithm fails to follow a desired order of operations for an input, it may fail to find a solution even though the solution exists.

`numerically stable`: Algorithms that can always find an appropriate operation order, thus finding a solution to the problem as long as it exists for any given input values, are called . 

` numerically unstable`: Algorithms that fall short are referred to as



例子：gaussian elimination需要使用pivioting的方法来解决numerical unstable的问题