# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import torch

def sum_demo(x,y):
    # 在下面的代码行中使用断点来调试脚本。
    # 按 Ctrl+F8 切换断点。
    for _ in range(2):
        x+=1
        y+=1;
    return x+y

if __name__=='__main__':
    result=sum_demo(1,1)
    print(result)
def qsort(ls):
    if len(ls)<=1:
        return ls
    else:
        pivot=ls[0]
        return qsort([x for x in ls[1:] if x<pivot])+[pivot]+qsort([x for x in ls[1:] if x>=pivot])
#写个函数判断一个数是否是素数
def is_prime(n):
    if n<2:
        return False
    else:
        for i in range(2,n):
            if n%i==0:
                return False
        return True
