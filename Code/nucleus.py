import numpy as np
import pandas as pd

from time import time
from os import listdir
from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import train_test_split

'''
    名称：get_pulse_data
    参数：filename：数列数据文件名，后缀是'.lis'的文件
    功能：解析序列数据，提取脉冲数据以及对应的时间
'''
def get_pulse_data(filename):
    fid = open(filename, 'rb')
    
    pulseheight = []
    realtime = []
    da1 = []
    da2 = []

    fid.seek( 256, 0 )
    data_array0 = np.fromfile(fid, dtype=np.uint32)

    RTword = np.where((data_array0 >= 2**31) & (data_array0 < (2**31 + 2**30)))
    RTwordcounts = RTword[0]

    fid.seek( 256, 0 )
    # with pbar.ProgressBar(max_value=RTwordcounts.size) as bar: 
    for i in tqdm(range(0, RTwordcounts.size)):
        fid.seek(RTwordcounts[i]*4, 1)
        da = np.fromfile(fid, dtype = np.uint16, count = 1)
        realtime10ms = da
        da1.append(da[0])

        fid.seek(14, 1)
        while 1:
            # data_array00 = np.fromfile(fid, dtype=np.uint32, count = 1)
            data_array1 = np.fromfile(fid, dtype=np.uint16, count=1)
            data_array2 = np.fromfile(fid, dtype=np.uint16, count=1)

            if data_array2 < 2**15 + 2**14:
                break
            realtime200ns = data_array1[0]
            # da2 = np.hstack((da2, realtime200ns))
            realtime0 = realtime200ns * 200 * 1e-6 + realtime10ms * 10  # ns 换算为ms
            realtime.append(realtime0)

            pulseheight.append(data_array2 - (2 ** 15 + 2 ** 14))
        fid.seek(256, 0)
    pulseheight = np.array(pulseheight)
    pulseheight = pulseheight.T
    realtime = np.array(realtime)
    realtime = realtime.T
    realtimepulse = np.hstack((realtime, pulseheight))
    
    return pulseheight, realtime, realtimepulse

'''
    名称：get_file_fulldata
    参数：fileName，文件的名称
    功能：一次全部获取数据文件夹中的数据
'''
def get_file_fulldata(fileAddress, split_len=2500, save_to_address='../Cs137data/numpy_data'):

    keyValue = {'BenDi': 0, 'Cs137': 1, 'CsCo': 2, 'Cs': 1, 'Co60': 3, 'Eu155': 4}

    start  = time()    
    nuc_index = []  # 核素的标签

    fileName = listdir(fileAddress)   # 获取文件夹目录
    pulseDataSet = np.zeros((1, split_len))
    realtimeDataSet = np.zeros((1, split_len))

    for i in range(len(fileName)):
        address = fileAddress + fileName[i]
        pulse, realtime, pr = get_pulse_data(address)

        saveAdd_pulse = save_to_address + 'pulse' + fileName[i].split('.')[0] + '.npy'
        saveAdd_realtime = save_to_address + 'realtime' + fileName[i].split('.')[0] + '.npy'
        np.save(saveAdd_pulse, pulse)
        np.save(saveAdd_realtime, realtime)

        timeData = np.zeros((realtime.shape[1] // split_len, split_len)) 
        pulseData = np.zeros((pulse.shape[1] // split_len, split_len))

        for j in range(realtime.shape[1] // split_len):
            pulseData[j, :] = pulse[0, j * split_len: j * split_len + split_len]
            timeData[j, :] = realtime[0, j * split_len : j * split_len + split_len]
            nuc_index.append(keyValue[fileName[i].split('_')[0]])
        pulseDataSet = np.vstack((pulseDataSet, pulseData))
        realtimeDataSet = np.vstack((realtimeDataSet, timeData))
        print('i = %d, len = %d, nuc is %s' % (i, len(timeData), fileName[i].split('_')[0]), end=' ')

    pulseDataSet = np.delete(pulseDataSet, 0, axis=0)        # 删除第一行
    realtimeDataSet = np.delete(realtimeDataSet, 0, axis=0)  # 删除第一行
    nuc_index = np.array(nuc_index)
    print('\n time is :%dm%ds' % ((time() - start) // 60, (time() - start) % 60))
    print('pulseDataSet.shape = ', pulseDataSet.shape,  'realtimeDataSet.shaep = ', 
            realtimeDataSet.shape, 'nuc_index.shape = ', nuc_index.shape)
    
    return pulseDataSet, realtimeDataSet, nuc_index


'''
   将数据转换成 CNN 训练的数据格式
   pulse: 脉冲数据
   realtime: 脉冲数据对应的实时时间
   labels: 数据标签，keyValue = {'BenDi': 0, 'Cs137': 1, 'CsCo': 2, 'Cs': 1, 'Co60': 3, 'Eu155': 4}
   Item_shape: 目标形状
'''
def sequential_data_trans_CNN_data(pulse, realtime, labels, Item_shape=[50, 50, 2]):
    # 创建一个随机索引
    rng = np.random.RandomState(2)
    indices = np.arange(pulse.shape[0])
    rng.shuffle(indices)
    # 打乱排序
    pulse = pulse[indices, :]
    realtime = realtime[indices, :]
    labels = labels[indices]
    # 删除时间循环点的数据集
    index = []
    for i in range(realtime.shape[0]):
        realtime[i, :] = realtime[i, :] - int(realtime[i, 0])
        if np.min(realtime[i]) < 0:
            index.append(i)
    index = np.array(index)
    print('len(index) = %d' % (len(index)))
    # 
    pulse = np.delete(pulse, index, 0)
    realtime = np.delete(realtime, index, 0)
    labels = np.delete(labels, index, 0)

    # 创建数据集
    dataSet = np.zeros((pulse.shape[0], Item_shape[0], Item_shape[1], Item_shape[2]))
    for i in range(pulse.shape[0]):
        for j in range(Item_shape[0]):
            dataSet[i, j, :, 0] = pulse[i, j * Item_shape[1]: j * Item_shape[1] + Item_shape[1]]
            dataSet[i, j, :, 1] = realtime[i, j * Item_shape[1]: j * Item_shape[1] + Item_shape[1]]
    
    print('pulse.shape = ', pulse.shape, 'realtime.shape = ', realtime.shape,
            'dataSet.shape = ', dataSet.shape)
    
    return dataSet, labels


'''
   函数名称：split_dataSet
   功能：划分数据集
'''
def split_train_test_dataSet(dataSet, labels, test_size, random_state):
    x_train, x_test, y_train, y_test = train_test_split(dataSet, labels, test_size=test_size, random_state=random_state)
    print('x_train.shape=', x_train.shape, 'y_train.shape=', y_train.shape,
            'x_test.shape=', x_test.shape, 'y_test.shape=', y_test.shape)
            
    return x_train, y_train, x_test, y_test

'''
    名称： predict_accuracy(result, y_test):
    功能：模型测试测试集数据集
    参数：result : 模型预测结果
          y_test ：测试集的正确标签
'''
def predict_accuracy(model, x_test, y_test):
    error_index = []
    num = 0
    result = model.predict(x_test)
    array = np.zeros((len(result), 1))
    for i in range(len(result)):
        array[i] = np.argmax(result[i])
    
    for i in range(len(array)):
        if array[i, 0] == y_test[i]:
            num += 1
        else:
            error_index.append(i)
    accuracy = num / len(y_test)
    print('accuracy = %f%%' % (accuracy * 100))

    return result, error_index, accuracy


def split_sequenceData(fileAddress, split_len):
    start = time()

    keyValue = {'pulseCs137': 0, 'pulseCsCo': 1, 'pulseCs': 0, 'pulseCo60': 2, 'pulseEu155': 3}
    fileName = listdir(fileAddress)
    
    nuc_index = []
    pulseDataSet = np.zeros((1, split_len))
    realtimeDataSet  = np.zeros((1, split_len))

    for i in range(len(fileName)):
        address = fileAddress + fileName[i]
        dataSet = np.load(address)
        
        if dataSet.shape[0] != 1:    # 直接跳出本次循环，不运行后续程序
            continue
        # saveAdd_realtime = '../Cs137data/numpy_data/' + 'realtime' + fileName[i].split('.')[0] + '.npy'
        # np.save(saveAdd_realtime, pulse )
        name = fileName[i].split('_')[0]

        if 'pulse' in name:
            pulseData = np.zeros((dataSet.shape[1] // split_len, split_len))
            for j in range(dataSet.shape[1] // split_len):
                pulseData[j, :] = dataSet[0, j * split_len: j * split_len + split_len]   
                nuc_index.append(keyValue[fileName[i].split('_')[0]])  # 标签设置   
            
            pulseDataSet = np.vstack((pulseDataSet, pulseData))   
            print('i = %d, len = %d, nuc is %s' % (i, len(pulseData), fileName[i].split('_')[0]) + '_' + fileName[i].split('_')[1])    

        elif 'realtime' in name:
            timeData = np.zeros((dataSet.shape[1] // split_len, split_len)) 
            for j in range(dataSet.shape[1] // split_len): 
                timeData[j, :] = dataSet[0, j * split_len : j * split_len + split_len]
            realtimeDataSet = np.vstack((realtimeDataSet, timeData))
            print('i = %d, len = %d, nuc is %s' % (i, len(timeData), fileName[i].split('_')[0]) + '_' + fileName[i].split('_')[1])            

    pulseDataSet = np.delete(pulseDataSet, 0, axis=0)        # 删除第一行
    realtimeDataSet = np.delete(realtimeDataSet, 0, axis=0)  # 删除第一行
    nuc_index = np.array(nuc_index)
    print('\n time is :%dm%ds' % ((time() - start) // 60, (time() - start) % 60))
    print('pulseDataSet.shape = ', pulseDataSet.shape,  'realtimeDataSet.shaep = ', 
            realtimeDataSet.shape, 'nuc_index.shape = ', nuc_index.shape)
    
    return pulseDataSet, realtimeDataSet, nuc_index



'''
pulseData = pd.read_csv('../Cs137data/csv/pulse2525.csv')
realtimeData = pd.read_csv('../Cs137data/csv/realtime2525.csv')

pulse = pulseData.values[:, 1: 2501]
realtime = realtimeData.values[:, 1: 2501]
labels = pulse[:, -1]

dataSet, labels = scquential_data_trans_CNN_data(pulse, realtime, labels, Item_shape=[50, 50, 2])
'''

# pulseDataSet, realtimeDataSet, nuc_index = split_sequenceData('../Cs137data/numpy_data/', split_len=900)