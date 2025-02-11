
## Requirements

Test Datasets: [Google Drive](https://drive.google.com/file/d/1r7Xaj8TuL_afy0svCtAfaXPqDmDWwuNR/view?usp=sharing)

Pretrained Models: [Google Drive](https://drive.google.com/file/d/13fHMg8DSLznjHqgB30khhP922EcbtpGG/view?usp=sharing)

## Dependencies

- Python 3.9
- Pytorch 2.1
- NVIDIA GPU + CUDA



```bash
git clone https://github.com/cnyvfang/Reti-Diff-demo.git
cd Reti-Diff-demo
conda create -n Reti-Diff python=3.9
conda activate Reti-Diff
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
python setup.py develop
```

### Install BasicSR
```bash
git clone https://github.com/xinntao/BasicSR.git
cd BasicSR
pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple
pip install -r requirements.txt
python setup.py develop
cd ..
```

## Usage

Put the Test Datasets into Datasets folder.

Put the Pretrained Models into pretrained_models folder.

Run the following command to test the model:

```bash
# LOLv2 Syn
sh test_LOLv2_Syn.sh

# UIEB
sh test_UIEB.sh

# BAID
sh test_BAID.sh
```