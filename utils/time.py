# -*- coding: utf-8 -*-
import datetime
import time

def get_result_dir():
    thisPath = get_now_time('%Y%m%d.%H%M%S')
    return thisPath

def get_now_time(fmt='%Y-%m-%d %H:%M:%S.%f'):
    """获取当前时间, %f表示毫秒"""
    nowTime = datetime.datetime.now().strftime(fmt)
    return nowTime

def sec_to_HMS(sec:float, separator=':'):
    """
    将秒转换为时分秒
    separator str | tuple | list
    """
    if type(separator) == str or len(separator) < 3:
        if type(separator) == list or type(separator) == tuple:
            separator = separator[0]
        separator = [separator for _ in range(2)]
        separator.append('')
    fmt = f"%H{separator[0]}%M{separator[1]}%S{separator[2]}"
    hms = time.strftime(fmt, time.gmtime(sec))
    return hms

