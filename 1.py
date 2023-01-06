# encoding: utf-8
'''
@author: lee
@license: (C) Copyright 2013-2022, Node Supply Chain Manager Corporation Limited. 
@contact: speakstone8875@gmail.com
@file: 1.py
@time: 2023/1/3 19:43
@desc:

'''
import os
import shutil
file_all = [i.split(".")[0] for i in os.listdir("/work/20221209_Aerial_Photograph/data/dota_/images")]
file_test = [i.split(".")[0] for i in os.listdir("/work/20221209_Aerial_Photograph/data/dota/val/labelTxt-v1.0/Val_Task2_gt/valset_reclabelTxt")]

with open("/work/20221209_Aerial_Photograph/data/dota_/test.txt", "w+") as f:
    for i in file_test:
        f.writelines(i + "\n")


with open("/work/20221209_Aerial_Photograph/data/dota_/trainval.txt", "w+") as f:
    for i in file_all :
        if i not in file_test:
            f.writelines(i + "\n")