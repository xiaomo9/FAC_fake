import torch
import torch.nn as nn
import multiprocessing
import time

def sing():
    for i in range(10):
        print("sing")
        time.sleep(1)

def dance():
    for i in range(10):
        print('dance')
        time.sleep(1)

process1 = multiprocessing.Process(target=sing)
process2 = multiprocessing.Process(target=dance)


if __name__ == '__main__':
    process1.start()
    process2.start()
    print(torch.cuda.is_available())