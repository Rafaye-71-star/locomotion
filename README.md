# Human Locomotion

## Configuration
```shell
# create virtual python environment
conda create -n nav python=3.7
# activate virtual environment
conda activate nav
# install dependence
pip install -r requirements.txt
# deactivate virtual environment
# conda deactivate nav
# remove virtual environment
# conda remove -n nav --all
```

## Run
Launch CarlaUE4.exe and run these script:
```shell
python Carla_Pedestrian_PPO.py
```

Run with GUI:
```shell
python Carla_Pedestrian_System_GUI.py
```
Select `Initialize Environment`, make sure the start and end points are in the csv, then click `Start Training`.





## Reference

* [Undergraduate thesis](https://github.com/OpenHUTB/sim/tree/master/pedestrian)


