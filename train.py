#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import sys

import numpy
from tqdm import tqdm
import common as com

import torch
import torch.nn as nn
import random

########################################################################


# In[2]:


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.
        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.
        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        #设置y轴范围
        #self.plt.ylim(-5,100)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.
        name : str
            save png file path.
        return : None
        """
        self.plt.savefig(name)


########################################################################


# In[3]:




def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset

def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files
    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir+dir_name))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files

def fetch_dataloaders(files,radio = 0.9,batchsize=512):
    len_d = len(files)
    num_sample_train = int(len_d * radio)
    index = list(range(len_d))
    random.shuffle(files)
    train = list(files[0:num_sample_train])
    val = list(files[num_sample_train+1:])
    train_data_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batchsize, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val, batch_size=batchsize, shuffle=True)
    loader = {}
    loader['train'] = train_data_loader
    loader['val'] = val_data_loader
    return loader


# In[5]:



########################################################################
#pytorch 模型相关
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(640, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 640),
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)

        return codes, decoded


# In[12]:



########################################################################
# 
########################################################################
if __name__ == "__main__":
    param = com.yaml_load()
    files = file_list_generator(param["train_dictionary"],dir_name="train")
    com.logger.info('files num {num}'.format(num=len(files)))
    train_data = list_to_vector_array(files,
                                        msg="generate train_dataset",
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])
    # train model
    print("============== MODEL TRAINING ==============")
    train_data = torch.Tensor(train_data)
    com.logger.info('train_data num {num}'.format(num=len(train_data)))
    train_loader = fetch_dataloaders(train_data,radio = 0.9,batchsize = 512)
    model = AutoEncoder()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    loss_his = []
    val_loss_his = []
    com.logger.info("epoch = {}".format(param["epoch"]))
    for epoch in range(param["epoch"]):
        losses=[]
        for data in tqdm(train_loader["train"]):
            optimizer.zero_grad()
            inputs = data
            feature,y = model(inputs)
            loss = loss_func(y, inputs)
            losses.append(loss.data)
            loss.backward()
            optimizer.step()
        loss_his.append(numpy.mean(losses))
        val_losses=[]
        for data in tqdm(train_loader["val"]):
            inputs = data
            feature,y = model(inputs)
            loss = loss_func(y, inputs)
            val_losses.append(loss.data)
            optimizer.step()
            optimizer.zero_grad()
        val_loss_his.append(numpy.mean(val_losses))
        com.logger.info("loss:{} epoch:{}".format(loss,epoch))

    
    torch.save(model, param["model_file"])
    one_visualizer = visualizer()
    one_visualizer.loss_plot(loss_his, val_loss_his)
    one_visualizer.save_figure(r'pic/train/'+"debug_tag{}_".format(str(param["debug_tag"]))+"trainloss_his.png")
    print("============== END TRAINING ==============")


# In[13]:



    features = []
    for data in tqdm(train_loader["train"]):
        inputs = data
        feature, y = model(inputs)
        for one_feature in feature:
            features.append(one_feature)
    numpy.save(param["normal_feature_dic"],numpy.array(features))

