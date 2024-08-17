# -*- coding: utf-8 -*-

# basic colors
BLACK = "\033[30m"
RED ="\033[31m"
GREEN ="\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
# bright colors
BRIGHT_BLACK = "\033[90m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"
# background color
BG_BLACK = "\033[40m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"
# misc
CLEAR = "\033[0m"
BOLD = "\033[1m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
REVERSE_DISPLAY = "\033[7m"
BLANKING = "\033[8m"  # 消隐

def colorstr(*args):
    """
    返回带颜色信息的字符串
    colorstr('blue','bold', 'underline', 123)
    """
    *args, string = args if len(args) > 1 else ("blue", "bold", args[0])  # color arguments, string
    colors = {
        # basic colors
        "black": BLACK,
        "red": RED,
        "green": GREEN,
        "yellow": YELLOW,
        "blue": BLUE,
        "magenta": MAGENTA,
        "cyan": CYAN,
        "white": WHITE,
        # bright colors
        "bright_black": BRIGHT_BLACK,
        "bright_red": BRIGHT_RED,
        "bright_green": BRIGHT_GREEN,
        "bright_yellow": BRIGHT_YELLOW,
        "bright_blue": BRIGHT_BLUE,
        "bright_magenta": BRIGHT_MAGENTA,
        "bright_cyan": BRIGHT_CYAN,
        "bright_white": BRIGHT_WHITE,
        # background color
        "bg_black": BG_BLACK,
        "bg_red": BG_RED,
        "bg_green": BG_GREEN,
        "bg_yellow": BG_YELLOW,
        "bg_blue": BG_BLUE,
        "bg_magenta": BG_MAGENTA,
        "bg_cyan": BG_CYAN,
        "bg_white": BG_WHITE,
        # misc
        "clear": CLEAR,
        "bold": BOLD,
        "italic": ITALIC,
        "underline": UNDERLINE,
        "reverse_display": REVERSE_DISPLAY,
        "blanking": BLANKING  # 消隐
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["clear"]

def print_color(k, *args, **kwargs):
    """
    k: list or tuple
        [123]
        ['red', 'italic', 'msg: hello world!!!']
        (123,)
    """
    ss = colorstr(*k)
    print(ss, *args, **kwargs)


if __name__ == '__main__':
    print_color(['red', 'underline', "\"123\": "], end='...0')
    print()
