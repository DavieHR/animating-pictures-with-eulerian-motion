#-*- coding: utf-8 -*-


# AUTHOR: wangharaon@glint.com
# TIME: 2020-07-13
# the python file include log system, face detection and landmark alignment class.

import time
import inspect
from functools import partial

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Glint_log(object):
    """ Glint Log print log with different color.
        RED ---> ERROR
        GREEN ---> DEBUG
        BLUE ---> INFO
        YELLOW --> WARNING.

        Attribute: 
            log_level: str, indicating the log level you used.
            log_blacklist: list, the log level you don't want to show, even though you write the log in certain file.
            _color_tag: tht color placeholder in print function.
    """
    def __init__(self, 
                 log_level,
                 log_blacklist = []):
        self._log_level = log_level.lower()
        self._color_tag = bcolors.HEADER
        
        if log_level not in log_blacklist:
            if log_level == 'warning':
                self._color_tag = bcolors.WARNING
            elif log_level == 'info':
                self._color_tag = bcolors.OKBLUE
            elif log_level == 'error':
                self._color_tag = bcolors.FAIL
            elif log_level == 'debug':
                self._color_tag = bcolors.OKGREEN
         
    def __call__(self, 
                *args, 
                **kwargs):
        _template_log = '{}[{}]'
        for arg in args:
            _template_log += '{} '
        _template_log += '{}'
        print(_template_log.format(self._color_tag, self._log_level.upper(), bcolors.ENDC, *args))



def Timer(func):
    """Timer Decorator.
    """
    def wrapper(*args, 
                **kwargs):
        """Timer wrapper.
        """
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        _function_name = inspect.currentframe().f_code.co_name
        Glint_log("info")("the function %s cost time is %f" % (_function_name, (end - start) / 1000.0))
    return wrapper
