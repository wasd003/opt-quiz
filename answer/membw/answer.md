- 使用STREAM benchmark,通过调整数组大小测试不同memory hierarchy上的内存带宽
- run.sh用来跑测试
    - 注意STREAM默认的是double数据类型，所以每个元素占8B
    - 最好结合上perf，验证确实内存访问都打到L1/L2/LLC/DRAM上
    - 测试L1/L2的时候，因为这些都是per cpu的，所以测试单核就可以了
    - 测试LLC/DRAM的时候，因为这些都是共享的，所以可以openmp开多线程

- 调整Makefile中的编译选项,指定数组大小
