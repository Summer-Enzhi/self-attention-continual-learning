import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
path_label = pd.read_csv('./SKIN8/ISIC_2019_Training_GroundTruth.csv')

source = "DF"
bkl_col = path_label[source]

path_label = path_label.drop(columns=[source])
path_label.insert(1, source, bkl_col)
path_label = path_label.iloc[:, :-1]
path_label.rename(columns={'image': 'path'}, inplace=True)
path_label['path'] = 'SKIN8/ISIC_2019_Training_Input/' + path_label['path'] + '.jpg'  # 修改路径

part_sizes = [10, 1, 1, 1, 1, 1, 1, 1]
parts = []
remaining_data = path_label.copy()

train_data, remaining_data = train_test_split(remaining_data, train_size=0.5, stratify=remaining_data.iloc[:, 1:], random_state=42)
valid_data, test_data = train_test_split(remaining_data, train_size=0.6, stratify=remaining_data.iloc[:, 1:], random_state=42)

train_total_length = train_data.shape[0]

for index,size in enumerate(part_sizes):
    if index==len(part_sizes)-1:
        continue
    size = int(train_total_length*(size/sum(part_sizes) ))
    print(size)
    part_data, train_data = train_test_split(train_data, train_size=size, stratify=train_data.iloc[:, 1:], random_state=42)
    parts.append(part_data)
parts.append(train_data)

for i, part in enumerate(parts):
    if i>0:
        part.iloc[:, 1:i+1] = 2
    if i<len(parts)-1:
        part.iloc[:, (i+2):] = 2

valid_data.iloc[:int(valid_data.shape[0]*10/17),2:]=2
test_data.iloc[:int(test_data.shape[0]*10/17),2:]=2

train_data  = pd.concat(parts)

train_data.iloc[:,1:] = train_data.iloc[:,1:].astype(int)
valid_data.iloc[:,1:] = valid_data.iloc[:,1:].astype(int)
test_data.iloc[:,1:] = test_data.iloc[:,1:].astype(int)

# 输出为csv文件
train_data.to_csv('skin8_train.csv', index=False)
valid_data.to_csv('skin8_valid.csv', index=False)
test_data.to_csv('skin8_test.csv', index=False)
