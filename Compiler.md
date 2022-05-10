


## Compiler



### optimize LLVM IR

> 参考
> 
> 1. MIT 6.172 l9

通过多轮的transformation来进行优化，passes run in predetermined order

Clang/LLVM可以report优化过程中的信息。但是很多transformation的信息并不会被打印,而且打印出来的信息比较难懂

```shell
-Rpass= -Rpass-missed -Rpass-analysis=*
```

<img src="Note.assets/Screen Shot 2021-10-14 at 6.31.18 PM.png" alt="Screen Shot 2021-10-14 at 6.31.18 PM" style="zoom:30%;" />

大部分的优化是在IR上面的，也会有部分在IR变为assembly上。




#### What Compiler Can Optimize

<img src="Note.assets/Screen Shot 2021-10-14 at 6.36.37 PM.png" alt="Screen Shot 2021-10-14 at 6.36.37 PM" style="zoom:30%;" />

<img src="Note.assets/Screen Shot 2021-10-14 at 6.36.53 PM.png" alt="Screen Shot 2021-10-14 at 6.36.53 PM" style="zoom:30%;" />

还有

1. eliminate dead code
2. strength some math computation : left shift insteaf of multiply by 2
3. reorder instruction to improve register reuse




#### optimize scalar

* leal

原本用来计算ram address，现在被用于进行简单的数学计算

```assembly
leal (,%rdi, 8), %eax
uint32_t x = n * 8;
```

* replace the stack-allocated variable with the copy in the register

O0：argument的变量会先申请stack，存放在stack，再从stack取

O1: 直接使用argument的register来进行计算，不分配stack，不存放stack

<img src="Note.assets/Screen Shot 2021-10-14 at 6.54.17 PM.png" alt="Screen Shot 2021-10-14 at 6.54.17 PM" style="zoom:30%;" />




#### optimize structure

O0: struct的信息不能都放在register上。需要先在stack上申请空间，把argument放入到stack上新分配的元素中。当使用的时候再unpack struct，从分配stack的变量中提取元素

<img src="Note.assets/Screen Shot 2021-10-14 at 6.57.20 PM.png" alt="Screen Shot 2021-10-14 at 6.57.20 PM" style="zoom:40%;" />

O1：optimize individual fields of aggregate type. 不再save&unpack struct, 直接使用argument中传入的variable

compiler transform data structure to store as much as possible in registers.

<img src="Note.assets/Screen Shot 2021-10-14 at 7.03.26 PM.png" alt="Screen Shot 2021-10-14 at 7.03.26 PM" style="zoom:40%;" />




#### optimize function calls

* function inlining 

对于很小的方程，直接inline，这样避免了funciton call overhead. 

还能在inline以后，进一步进行优化(addisional transformation)，例如很多struct的pack&unpack

缺点

1. recursion call无法被inline
2. 无法inline另一个单独编译单元中的代码，除非使用whole-program optimization
3. 增加code size

programmer控制

1. always inline
2. never inline
3. Link-time optimization LTO




#### optimize loops

loop占用了execution的很多时间。

* hoisting: loop-invariant code optimization (LICM)。把一些可以在loop外面计算的从loop中移出去，从而减少总的instruciton数量
* unroll ：compiler 会帮助进行unroll，根据compiler判断的loop body资源使用情况
* fuses loop : 从而减少control branch

* vectorization

compiler可以分析潜在的vectorzation。

一般assembly会分为vectorization版本和非vectorization版本。之所以有两个版本是因为不确定input的pointer是否可能指向同一块内存。如果使用了restrict的话，就只有vectorize的版本

如果希望compiler分析loop来vectorize的话，**需要设定i为signed type**。因为unsigned type可能会overflow到0，当使用unsigned type的时候compiler不会向量化。在C中，尽管signed interger也会overflow，但是结果是undefined的，所以compiler会假设signed integer永远不会overflow。

```cpp
for ( int64_t i = 0; i < n; i += 4 )
{
  z[i] = a * x[i] + y[i];
  z[i+1] = a * x[i+1] + y[i+1];
  z[i+2] = a * x[i+2] + y[i+2];
  z[i+3] = a * x[i+3] + y[i+3];
}
```




#### link-time optimization (LTO)

compiler only transform code within a single file ( compilation unit )

对于每一个文件，首先生成LLVM IR，汇聚文件进行优化，然后再进行优化。

```shell
-flto # compile source code to llvm bitcode
-flto -fuse-ld=gold # gold linker to link llvm vitcode files together
```




### GNU GCC




#### attribute

> 参考
> 
> 1. https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/Function-Attributes.html#Function-Attributes
> 2. MIT 6.172 l9

适用于函数的声明，而不是定义。

一种编译器的进阶优化

* 开启GNU attribute

```cpp
#ifndef __GNUC__
    #define __attribute(x) /* NOTHING */
#endif
```

##### const

方程return只与input相关，同时方程不会更改global memory。使用const标注方程以后使得同一input，只会计算一次。实现common-subexpression elimination

```cpp
__attribute__((const))
double norm(const double*x, int n)
{
    return xxx;
}
```

##### aligned

指定struct按照某一个指定大小进行对齐

```cpp
struct s
{
  short f[3];
} __attribute__((aligned(8)));
```

##### packed

指定struct不进行内存对齐，而是尽量使用最小内存

```cpp
struct
{
  int a;
  double b;
  char c;
} __attribute__((packed));
```

