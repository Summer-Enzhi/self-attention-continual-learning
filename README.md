# Enhancing Continual Learning for Medical Imaging: Efficient Knowledge Transfer and Multi-Disease Prediction

## How to Use

### Prepare environment

```bash
conda create -n CL_Pytorch python=3.8
conda activate CL_Pytorch
pip install -r requirement.txt
```
### Prepare data
`AIROGS` : https://airogs.grand-challenge.org/data-and-challenge/

`RFMID`  : https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid

`SKIN8`  : https://www.kaggle.com/datasets/kioriaanthony/isic-2019-training-input

Put them all in a specified folder x, then specify the path to x in `os.environ['DATA']`. Then execute the preprocessing script in `./preprocess`.

### Run experiments

1. Edit the hyperparameters in the corresponding `options/XXX/XXX.yaml` file

2. Train models:

```bash
python main.py --config options/XXX/XXX.yaml
```

3. Test models with checkpoint (ensure save_model option is True before training)

```bash
python main.py --checkpoint_dir logs/XXX/XXX.pkl
```


Our run script is in the `./bash` folder. If you want to temporary change GPU device in the experiment, you can type `--device #GPU_ID` in terminal without changing 'device' in `.yaml` config file.

### Add datasets and your method

Add corresponding dataset .py file to `datasets/`. It is done! The programme can automatically import the newly added datasets.

we put continual learning methods inplementations in `/methods/multi_steps` folder, pretrain methods in `/methods/pretrain` folder and normal one step training methods in `/methods/singel_steps`.

Supported Datasets: Skin8, AIRGOS-RFMID

More information about the supported datasets can be found in `datasets/`

We use `os.environ['DATA']` to access image data. You can config your environment variables in your computer by editing `~/.bashrc` or just change the code.


## References

https://github.com/GiantJun/CL_Pytorch