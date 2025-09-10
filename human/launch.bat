
:: conda activate locomotion

:: 模型路径：D:/work/workspace/locomotion/human/
set MODEL_PATH=D:/work/workspace/locomotion/human/

:: 训练
:: 走路
python puppeteer/train.py task=tracking low_level_fp=%MODEL_PATH%model/tracking.pt




:: 训练非视觉任务
python puppeteer/train.py task=stand low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt
python puppeteer/train.py task=walk low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt
python puppeteer/train.py task=run low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt


:: 验证 3 个非视觉任务：stand、walk、run

:: 验证站立
:: python puppeteer/evaluate.py task=stand low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/logs/stand/1/default/models/1500000.pt save_video=true

:: 验证走路
:: python puppeteer/evaluate.py task=walk low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/logs/walk/1/default/models/1000000.pt save_video=true

:: 验证跑步
python puppeteer/evaluate.py task=run low_level_fp=%MODEL_PATH%model/tracking.pt checkpoint=%MODEL_PATH%logs/run/1/default/models/1500000.pt save_video=true
:: python puppeteer/evaluate.py task=run low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/logs/run/1/default/models/1500000.pt save_video=true



:: 验证 5 个视觉任务

:: 过道 corridor（正常：corridor-10.mp4）
python puppeteer/evaluate.py task=corridor low_level_fp=%MODEL_PATH%model/tracking.pt checkpoint=%MODEL_PATH%model/corridor-3.pt save_video=true
:: python puppeteer/evaluate.py task=corridor low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/model/corridor-3.pt save_video=true


:: 跨栏过道（正常：hurdles-corridor-10.mp4）
:: 视频保存在：locomotion\human\logs\hurdles-corridor\1\default\videos\
python puppeteer/evaluate.py task=hurdles-corridor low_level_fp=%MODEL_PATH%model/tracking.pt checkpoint=%MODEL_PATH%model/hurdles-3.pt save_video=true


:: 绕墙过道（正常：walls-corridor-9.mp4）
python puppeteer/evaluate.py task=walls-corridor low_level_fp=%MODEL_PATH%model/tracking.pt checkpoint=%MODEL_PATH%model/walls-3.pt save_video=true


:: 沟渠过道（正常：gaps-corridor-19.mp4）
python puppeteer/evaluate.py task=gaps-corridor low_level_fp=%MODEL_PATH%model/tracking.pt checkpoint=%MODEL_PATH%model/gaps-3.pt save_video=true


:: 楼梯过道 stairs-3.pt（后面摔跤：stairs-corridor-18.mp4）
python puppeteer/evaluate.py task=stairs-corridor low_level_fp=%MODEL_PATH%model/tracking.pt checkpoint=%MODEL_PATH%model/stairs-3.pt save_video=true

