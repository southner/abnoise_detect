import os
import glob
import sys

import common as com
import numpy
import tqdm
import sklearn
import torch
from train import list_to_vector_array
from train import file_list_generator
from train import FCNet


def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
    com.logger.info("target_dir : {}".format(target_dir + "_" + id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")
        print("\n========================================")
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels


if __name__ == '__main__':
    # load data
    param = com.yaml_load()
    dirs = com.select_dirs(param=param, mode=False)  # eval
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue

        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        files = file_list_generator(target_dir)
        eval_data = list_to_vector_array(files,
                                         msg="generate eval_dataset",
                                         n_mels=param["feature"]["n_mels"],
                                         frames=param["feature"]["frames"],
                                         n_fft=param["feature"]["n_fft"],
                                         hop_length=param["feature"]["hop_length"],
                                         power=param["feature"]["power"])

    eval_data = torch.Tensor(eval_data)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_data, batch_size=512, shuffle=False)
    model = torch.load('model/model.pkl')
    model.eval()
    # test
    loss_func = torch.nn.MSELoss()
    for data in eval_loader:
        inputs = data
        y = model(inputs)
        loss = loss_func(y, inputs)
        print(loss)

