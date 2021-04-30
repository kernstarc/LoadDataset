'''
生成训练和测试，保存txt文件中
'''
import os
from os import getcwd   # 方法用于返回当前工作目录

train_rate = 1
test_rate = 1 - train_rate

roots = getcwd()   # 获取当前绝对路径
roots += "/FlowersData/data/"

print("Datasets root:", roots)

class_flag = -1

train_txt = open("train.txt", "w", encoding='utf-8')
test_txt  = open("test.txt", "w", encoding='utf-8')

# 用于通过在目录树中游走输出在目录中的文件名
'''
roots:  代表需要遍历的根文件夹
root:   表示正在遍历的文件夹的名字（根/子）
dirs:   记录正在遍历的文件夹下的子文件夹集合
files:  记录正在遍历的文件夹中的文件集合
'''
train_data_size = 0
test_data_size  = 0
for root, dirs, files in os.walk(roots):   
    print(root)
    for i in range(0, int(len(files)*train_rate)):
        # 用于路径拼接文件路径
        train_txt.write(os.path.join(root, files[i]) + '\t' + str(class_flag) + '\n')
        train_data_size += 1

    for j in range(int(len(files)*train_rate), len(files)):
        test_txt.write(os.path.join(root, files[i]) + '\t' + str(class_flag) + '\n')
        test_data_size += 1

    class_flag = class_flag + 1

print("train data size:%d" % train_data_size)
print("test data size :%d" % test_data_size)
print("Finish to process data img\nall have %d classes" % class_flag)



