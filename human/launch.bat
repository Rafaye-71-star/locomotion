
conda activate locomotion

:: 站立


:: 过道 corridor
python puppeteer/evaluate.py task=corridor low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/model/corridor-1.pt save_video=true


python evaluate.py task=gaps-corridor low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/model/gaps-1.pt save_video=true


:: 楼梯过道 stairs-3.pt
python puppeteer/evaluate.py task=stairs-corridor low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/model/stairs-3.pt save_video=true

