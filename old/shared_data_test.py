from SharedData import SharedNumpyArray, SharedPandasDataFrame
import numpy as np

shared_arr = SharedNumpyArray(np.ndarray(shape=(10000,)))

arr = shared_arr.read()
for i in range(100):
    arr[i] = i

arr1 = shared_arr.read()
for i in range(100):
    print(arr1[i])

shared_arr.unlink()
shared_arr = SharedNumpyArray(arr1[:10])

arr2 = shared_arr.read()
for i in range(10):
    print(arr2[i])

shared_arr.unlink()
