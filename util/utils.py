import time
import socket
import os
import numpy as np
import torch
import random
import logging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

def get_time():
    return time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())

def get_hostname():
    return socket.gethostname()

def get_pid():
   return os.getpid()

# set random number generators' seeds
def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import logging
logger_init = False

def init_logger(_log_file, use_file_logger=True, dir='log/'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    log_file = os.path.join(dir, _log_file + '.log')
    #logging.basicConfig(filename=log_file, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',level=logging.DEBUG)
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    logger.addHandler(chlr)
    if use_file_logger:
        fhlr = logging.FileHandler(log_file)
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)

    global logger_init
    logger_init = True