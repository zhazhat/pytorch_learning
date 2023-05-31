def qsort(arr):
    if len(arr)<=1:
        return arr
    else:
        pivot=arr[0]
        return qsort([x for x in arr[1:] if x<pivot])+[pivot]+qsort([x for x in arr[1:] if x>=pivot])
#判断是否为素数
def is_prime(n):
    if n<2:
        return False
    else:
        for i in range(2,n):
            if n%i==0:
                return False
        return True
#冒泡排序
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1-i):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
#用pytorch写个神经网络
