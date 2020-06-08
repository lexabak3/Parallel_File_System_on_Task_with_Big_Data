from shutil import copyfile
import os
import re

n_copy = 10000

path = os.getcwd() + '/cifar-10-batches-py'
ls_dir = os.listdir(path)
max_num = max([int(re.search(r'[\d]+', x).group()) for x in ls_dir if re.search(r'[\d]+', x) is not None])
print(max_num)
for n in range(n_copy):
    for i in range(1, 6):
        # f_path = os.path.join(path, 'data_batch_' + str(i))
        # copyfile(f_path, f_path[:-1] + str(max_num+1))
        max_num += 1
print(max_num)


