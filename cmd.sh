#!/bin/bash
# encoding: UTF-8

# . script/bash/git.sh
source script/bash/git.sh

arg=$1
arg2=$2
arg3=$3

help(){
    if [[ $arg != '' && $arg != 'help' && $arg != 'h' && $arg != '-h' ]];then
        echo "  未知命令: " $arg
    fi
    echo -e \
        "  Usage:\n"\
        "\tbash cmd.sh add file [message]           执行: git add one file and commit\n"\
        "\tbash cmd.sh all [message]                执行: git add . and commit\n"\
        "\tbash cmd.sh push, add_push [message]     执行: git add, commit and push\n"\
        "\tbash cmd.sh init                         执行: git init and first commit\n"\
        "\tbash cmd.sh user [username] [email]      执行: git config user\n"\
        "\tbash cmd.sh conf_list                    执行: git config --list\n"\
        "\tbash cmd.sh reset_cache                  执行: git rm cache 更新仓库管理规则\n"\
        "\tbash cmd.sh h, -h, help                  执行: help information"
}

if [[ $arg == 'add' ]];then
    filename=$arg2
    msg=$arg3
    add $filename "$msg"
elif [[ $arg == 'all' ]];then
    add_all "$arg2"
elif [[ $arg == 'push' || $arg == 'add_push' ]];then
    add_push "$arg2"
elif [[ $arg == 'init' ]];then
    init
elif [[ $arg == 'user' ]];then
    config_user $arg2 $arg3
elif [[ $arg == 'conf_list' ]];then
    config_list
elif [[ $arg == 'reset_cache' ]];then
    reset_cache
elif [[ $arg == 'help' || $arg == 'h' || $arg == '-h' ]];then
    help
else
    help
fi