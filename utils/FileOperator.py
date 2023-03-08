import glob
import logging
import os
import time
from os.path import splitext
from tqdm import tqdm
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def auto_make_directory(dir_pth: str):
    '''
    自动检查dir_pth是否存在，若存在，返回真，若不存在创建该路径，并返回假
    :param dir_pth: 路径
    :return: bool
    '''
    if os.path.exists(dir_pth):  ##目录存在，返回为真
        return True
    else:
        os.makedirs(dir_pth)
        return False
def get_dirs_pth(dir_pth: str):
    '''
    返回返回dir_pth下文件夹路径
    :param dir_pth:
    :return: 文件夹绝对路径list
    '''
    rst = []
    for item in os.listdir(dir_pth):
        temp = os.path.join(dir_pth, item)
        if os.path.isdir(temp):
            rst.append(str(temp))
    return rst


def get_dirs_name(dir_pth: str):
    rst = []
    for item in os.listdir(dir_pth):
        temp = os.path.join(dir_pth, item)
        if os.path.isdir(temp):
            rst.append(str(item))
    return rst
def get_filename_from_pth(file_pth: str, suffix: bool = True):
    '''
    根据文件路径获取文件名
    :param file_pth:文件路径
    :return:文件名
    '''
    fname_list = os.path.split(file_pth)[1].split('.')
    if suffix: #如果保留后缀

        rst = '.'.join(fname_list)
    else:#如果不保留后缀
        rst = '.'.join(fname_list[:-1])
    return rst

def get_files_pth(dir_pth: str, suffix: str = '*'):
    '''
    返回dir_pth下以后缀名suffix结尾的文件绝对路径list
    :param dir_pth:文件夹路径
    :param suffix:限定的文件后缀
    :return: 文件绝对路径list
    '''
    rst = []
    glob_pth = os.path.join(dir_pth, f'*.{suffix}')
    for filename in glob.glob(glob_pth):
        rst.append(filename)
    return rst


import csv
def write_csv(rst: list, file_pth: str, overwrite=False):
    '''
    :param rst:形如[('val1', val2),...,('valn', valn)]的列表
    :param file_pth:输出csv的路径
    :return:
    '''
    mode = 'w+' if overwrite else 'a+'
    file = open(file_pth, mode, encoding='utf-8', newline='')

    csv_writer = csv.writer(file)

    csv_writer.writerows(rst)

    file.close()
def get_all_files_pth(dir_pth: str, suffix: str = None):
    '''
    获取指定文件夹下（含子目录）以指定后缀结尾的文件路径列表
    :param dir_pth: 指定文件夹路径
    :param suffix: 指定后缀
    :return:
    '''
    rst = []
    for root, dirs, files in os.walk(dir_pth):
        if len(files) > 0:
            for file_name in files:
                file_pth = os.path.join(root, file_name)
                if not suffix:
                    rst.append(file_pth)
                elif file_pth.endswith(f'.{suffix}'):
                    rst.append(file_pth)
    return rst


def get_files_name(dir_path: str, suffix: str = '*'):
    '''
    返回指定文件夹内的文件名（不带后缀）列表
    :param dir_path: 文件夹路径
    :param suffix:限定的文件后缀
    :return:文件名（不带后缀）列表
    '''
    if suffix == '*':
        ids = [splitext(file)[0] for file in os.listdir(dir_path) if not file.startswith('.')]
        return ids
    else:
        ids = [splitext(file)[0] for file in os.listdir(dir_path) if file.endswith(f'.{suffix}')]  # 获取图片名称，ids是一个列表
        return ids


def get_filename_from_pth(file_pth: str, suffix: bool = True):
    '''
    根据文件路径获取文件名
    :param file_pth:文件路径
    :return:文件名
    '''
    fname_list = os.path.split(file_pth)[1].split('.')
    if suffix: #如果保留后缀

        rst = '.'.join(fname_list)
    else:#如果不保留后缀
        rst = '.'.join(fname_list[:-1])
    return rst


def get_suffix_from_pth(file_pth: str):
    '''
    根据文件路径获取后缀
    :param file_pth:文件路径
    :return:后缀
    '''
    return os.path.split(file_pth)[1].split('.')[-1]


import os, re


def remove_strip(dir,sign:str=' '):
    '''
    去除指定路径dir下文件名中的空格，也可以去除其它字符、符号
    :param dir:
    :return:
    '''
    files = get_files_pth(dir)
    for pth in tqdm(files):
        os.rename(pth, pth.replace(sign, ''))

# remove_strip(r'C:\Users\Ocean\Desktop\panel\mask','_')
# print(splitext(r'C:\Users\Ocean\Desktop\panel\mask\a.png'))
def append_suffix(dir,suffix:str):
    files = get_files_pth(dir)
    for pth in tqdm(files):
        os.rename(pth,splitext(pth)[0]+suffix+splitext(pth)[1])

def rename_pathes(dir):
    files = get_files_pth(dir)
    for pth in tqdm(files):
        new_pth = pth.replace('芯片隐裂','crack')
        print(new_pth)
        os.rename(pth,new_pth)
# rename_pathes(r'D:\hongpu\数据集\crack\原图')
# append_suffix(r'C:\Users\Ocean\Desktop\panel\bbp_mask',r'_0')
# s = r'LRP9041062106009001130_1.png'
# print(splitext(s)[0].split('_')[-1])
def purge(dir, pattern):
    '''
    根据正则表达式规则pattern清洗dir下文件，文件名不符合pattern的文件将被删除
    :param dir:
    :param pattern:
    :return:
    '''
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def get_common_file_in_dir(dir_a, dir_b):
    '''
    对dir_a，dir_b中的文件清晰，忽略后缀名，仅保留两个文件夹共有的文件
    :param dir_a:
    :param dir_b:
    :return:
    '''
    a_fnames = get_files_name(dir_a)
    b_fnames = get_files_name(dir_b)
    a_set = set(a_fnames)
    b_set = set(b_fnames)
    diff = list(a_set.symmetric_difference(b_set))
    if len(a_set) < len(b_set):  # a为基准 删除b
        del_dir = dir_b
    else:
        del_dir = dir_a
    for del_fname in diff:
        cur_fpth = os.path.join(del_dir, del_fname)
        cur_fpth = glob.iglob(cur_fpth + '.*')
        for i in cur_fpth:
            print(f'删除{i}')
            os.remove(i)
    print(f'共删除{len(diff)}个样本')

# get_common_file_in_dir(r'C:\Users\Ocean\Desktop\panel\bp_mask',r'C:\Users\Ocean\Desktop\panel\bp_img')
