import pandas as pd

RFMID= pd.read_csv('./RFMID/Training_Set/RFMiD_Training_Labels.csv',encoding='unicode_escape')
select = RFMID.iloc[:,2:9]
select = pd.concat([select,RFMID.iloc[:,13]],axis=1)
for j in range(select.shape[1]):
    print(select.iloc[:,j].value_counts())
    
RFMID_columns = ['ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ODC']
SKIN8_columns = ['NV', 'BCC', 'AK', 'BKL', 'VASC', 'SCC']
for i in range(select.shape[1]):
    if i!=0:
        select.iloc[:int(i*select.shape[0]/8),i]=2
    select.iloc[int((i+1)*select.shape[0]/8):,i]=2


select.insert(0,column='ID',value=RFMID.iloc[:,0])
for i in range(RFMID.shape[0]):
    select.loc[i,'ID']=f"RFMID/Training_Set/Training/{select.loc[i,'ID']}.jpg"
select.head()
select.insert(1,'AIROGS',2)

data = pd.read_csv(f'AIROGS/train_labels.csv')
for i in range(data.shape[0]):
    prefix = f'AIROGS/{i//18000}/'
    data.iloc[i]['challenge_id'] = prefix+data.iloc[i]['challenge_id']+'.jpg'
    if data.iloc[i]['class']=='NRG':
        data.iloc[i]['class']=0
    else:
        data.iloc[i]['class']=1
        
columns = data.columns
new_data = pd.DataFrame(2, index=data.index, columns=range(len(columns), len(columns) + 8))

# 将新数据连接到原始数据上
data = pd.concat([data, new_data], axis=1)
data.columns = ['path']+list(range(9))
# 1. 筛选出第 0 类为 1 的行
selected_rows_class_1 = data[data[0] == 1]

# 2. 从第 0 类为 0 的行中选择与第 0 类为 1 的行数量相等的两倍数量的行
selected_rows_class_0 = data[data[0] == 0].sample(n=len(selected_rows_class_1)*1, random_state=42) #第二次：n=len(selected_rows_class_1)*5

# 3. 合并这两部分数据成新的数据集 data2
data = pd.concat([selected_rows_class_1, selected_rows_class_0])

data = data.sample(frac=1, random_state=42)
# 重新索引以保持连续索引值
data.reset_index(drop=True, inplace=True)

select.columns = data.columns
from sklearn.model_selection import train_test_split

# 假设你的 DataFrame 名称是 data
# train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# train_data 是训练集，test_data 是测试集
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)
train_data.head()

alldata = pd.concat([train_data,select],axis=0)
# alldata = select
alldata.info()

alldata.to_csv('AR_train.csv',index=None)

RFMIDt= pd.read_csv('./RFMID/Evaluation_Set/RFMiD_Validation_Labels.csv',encoding='unicode_escape')
selectt = RFMIDt.iloc[:,2:9]
selectt = pd.concat([selectt,RFMIDt.iloc[:,13]],axis=1)

selectt.insert(0,column='ID',value=RFMIDt.iloc[:,0])
for i in range(RFMIDt.shape[0]):
    selectt.loc[i,'ID']=f"RFMID/Evaluation_Set/Validation/{selectt.loc[i,'ID']}.jpg"
selectt.head()
selectt.insert(1,'AIROGS',2)
selectt

selectt.columns = test_data.columns


alldata_test = pd.concat([test_data,selectt],axis=0)
# alldata_test = selectt
# alldata_test.to_csv('AR_test.csv',index=None)
# vali, test = train_test_split(alldata_test, test_size=0.5, random_state=42)
test, vali = train_test_split(alldata_test, test_size=0.5, random_state=42)

# train_data 是训练集，test_data 是测试集
vali.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)


vali.to_csv('AR_validation.csv',index=None)
test.to_csv('AR_test.csv',index=None)
