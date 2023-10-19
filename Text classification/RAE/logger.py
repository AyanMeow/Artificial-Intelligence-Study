import logging
import time
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    """docstring for Logger."""
    def __init__(self,) :
        super(Logger, self).__init__()
        self.filepath='./Text classification/RAE/log'
        name=time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime()) 
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(self.filepath+'/'+name+'.txt')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s  - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.tbpath='./tensorboard'
        self.writer = SummaryWriter(log_dir=self.tbpath, filename_suffix=name,flush_secs = 180)

    def newlog(self):
        name=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(self.filepath+'/'+name+'.txt')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def newtb(self):
        name=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 
        self.writer=SummaryWriter(log_dir=self.tbpath, filename_suffix=name,flush_secs = 180)
    
    def logout(self,logstr:str):
        self.logger.info(logstr)
    

    