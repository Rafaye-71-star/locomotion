
conda activate locomotion

python puppeteer/evaluate.py task=corridor low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/model/corridor-1.pt save_video=true


python evaluate.py task=gaps-corridor low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/model/gaps-1.pt save_video=true
