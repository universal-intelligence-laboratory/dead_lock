# dead_lock
## 笔记
普通的卷积的卷积的步骤是：
- 在输入的第一个通道上，拿k*k的kernel做卷积，产生一个同样大小的单通道feature map
- 在每个通道上做同样的事，产生Cin个feature map，叠加并得到一个单通道feature map
- 重复上面的Cout次，每次得到的单通道feature map，都加上一个bias，从而获得Cout个feature map（如果bias设为0，相当于只是把输出复制了n份）

我加上一步：  
- 对于Cout个feature map，每个feature map * 一个scalar
- 训练时初始化为1并固定训练
- 裁剪的时候固定其他网络参数，允许这个scalar进行调节，但一旦这个scalar接近0，马上置零并停止梯度
- inferrance的时候，只计算非0的对应的kernel

## 笔记
- 在普通卷积之前，检查死亡数目并只在存活的channel上做卷积
- 在乘dead之前，加一个极小值死亡函数
