@echo off

set python=E:\A_SoftwareInstall\python\python.exe
set python_cpu=E:\A_SoftwareInstall\anaconda3\envs\flow\python.exe
set python_gpu=E:\A_SoftwareInstall\anaconda3\envs\flow_cuda\python.exe

if "%1" == "1" ( rem cpu
    E:\A_SoftwareInstall\anaconda3\envs\flow\python.exe .\script\press_cnn\train.py -d v4_wms
    @REM %python_cpu% .\script\press_cnn\train.py -d v4_wms
) else if "%1" == "2" ( @REM cuda, gpu
    E:\A_SoftwareInstall\anaconda3\envs\flow_cuda\python.exe .\script\press_cnn\train.py -d v4_wms
    @REM %python_gpu% .\script\press_cnn\train.py -d v4_wms
) else if %1 == 3 (
    E:\A_SoftwareInstall\anaconda3\envs\flow_cuda\python.exe .\script\press_cnn\train.py -d v4_diffPress
) else if %1 == 4 (
    E:\A_SoftwareInstall\anaconda3\envs\flow_cuda\python.exe .\script\press_cnn\train.py -d v4_press4
) else if %1 == 5 (
    E:\A_SoftwareInstall\anaconda3\envs\flow_cuda\python.exe .\script\press_cnn\train.py -d v4_press4  --length 2048
)