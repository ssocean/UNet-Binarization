import glob
import logging
import os
import time
from os.path import splitext


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


def init_logger(out_pth:str='logs'):
    '''
    初始化日志类
    :param out_pth: 输出路径，默认为调用文件的同级目录logs
    :return: 日志类实例对象
    '''
    # 日志模块
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    auto_make_directory(out_pth)
    handler = logging.FileHandler(fr'{out_pth}/{time.strftime("%Y_%b_%d", time.localtime())}_log.txt')

    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 输出到日志
    logger.addHandler(handler)
    logger.addHandler(console)
    '''
    logger = init_logger(r'r')
    logger.info("Start print log") #一般信息
    logger.debug("Do something") #调试显示
    logger.warning("Something maybe fail.")#警告
    logger.info("Finish")
    '''
    return logger

def get_dirs_pth(dir_pth: str):
    '''
    返回返回dir_pth下文件夹路径
    :param dir_pth:
    :return: 文件夹绝对路径list
    '''
    rst = []
    for item in os.listdir(dir_pth):
        temp = os.path.join(dir_pth,item)
        if os.path.isdir(temp):
            rst.append(str(temp))
    return rst

def get_dirs_name(dir_pth: str):
    rst = []
    for item in os.listdir(dir_pth):
        temp = os.path.join(dir_pth,item)
        if os.path.isdir(temp):
            rst.append(str(item))
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


def get_filename_from_pth(file_pth: str, suffix: bool=True):
    '''
    根据文件路径获取文件名
    :param file_pth:文件路径
    :return:文件名
    '''
    fname_list = os.path.split(file_pth)[1].split('.')
    rst = '.'.join(fname_list)
    return rst


def get_suffix_from_pth(file_pth: str):
    '''
    根据文件路径获取后缀
    :param file_pth:文件路径
    :return:后缀
    '''
    return os.path.split(file_pth)[1].split('.')[1]

