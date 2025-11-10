import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

text_file_name = "big-loop-output.txt"

file_path = os.path.join(parent_dir, text_file_name)

file_list = []

with open(file_path, 'r') as infile:
    for line in infile:
        templine = [el for el in line.split(' ')]
        if len(templine) > 19:
            file_list.append(templine)


i=0
accurate_list = []
#accurate_list = [i for i in file_list[i][10] if float(list(file_list[i][10].pop(-1))) > 0.9]


for i in range(len(file_list)):
    temp_fl = file_list[i][10]
    temp_fl = float(temp_fl[:-1])
    if temp_fl > 0.974:
        accurate_list.append(file_list[i])
    
for line in accurate_list:
    print(line)

# from collections import defaultdict

# dict1 = defaultdict(list)
# key = []
# for line in accurate_list:
#     accuracy = line[10]
#     activation = line[14]
#     opt = line[17]
#     key.append(activation)
#     key.append(opt)
#     dict1[accuracy] = key

# vals = dict1.values()
# for i,v in enumerate(vals):
#     acc = 
#     print(f"Activation function: {vals[0]} with accuracy. Optimizer: {vals[1]}")