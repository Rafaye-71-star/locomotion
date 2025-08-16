:: 创建虚拟Python环境
conda env list | findstr /i "locomotion"
if %errorlevel% neq 0 (
    echo "Creating new conda environment 'locomotion'..."
    call conda create -n locomotion -c conda-forge python=3.10 pip ipython cudatoolkit=11.8.0 --yes
) else (
    echo "Conda environment 'locomotion' already exists."
)
:: 激活虚拟环境
call conda activate locomotion
:: 安装依赖
pip install -r requirements.txt

pip install git+https://github.com/deepmind/acme.git
pip install git+https://github.com/TuragaLab/flybody.git

conda install -c conda-forge ffmpeg
