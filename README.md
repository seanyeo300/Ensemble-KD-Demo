<h1>Knowledge Distillation with PaSST to BCBL</h1>

<h2>Description</h2>


This repository consists of a simple approach to train a low-complexity ASC model using Knowledge Distillation. The teacher model uses the Patchout faSt Spectrogram Transformer ([PaSST](https://arxiv.org/abs/2110.05069)) [1] model pre-trained with AudioSet [2]. The student model uses the DCASE2024 Baseline model.
<br/>

The training dataset is the TAU 2022 urban acoustic scenes mobile development dataset. The h5 files for the training dataset and dir augmentations are provided. 

**For the DCASE challenge, you will need to adjust the eval_dl dataloader and dataset functions to load the evaluation data from wav files.**


<br />


<h2>Languages and Utilities Used</h2>

The important packages and libraries required for reproducing this experiment are:

- <b>Python = 3.7.9 </b> 
- <b>[Pytorch](https://pytorch.org/get-started/previous-versions/) = 1.13.1+cu117 </b>
- <b>pytorch-lightning = 1.9.5 </b>
- <b> timm =0.4.12 </b>
<h2>Environments Used </h2>

A good place to build your environment is by installing the requirements from [here](https://github.com/CPJKU/dcase2024_task1_baseline).
**Do not upgrade your lightning version**

- <b>Windows 10</b>

<h2>Datasets downloaded</h2>

- <b>TAU 2022 Urban Acoustic Scenes Mobile Development dataset [3] </b> 

<h2>Getting Started:</h2>

1. Set up your WANDB account and obtain your [API key](https://docs.wandb.ai/quickstart/)
2. Login using the command line. You can now monitor your experiments in real time. I recommend setting up email updates for run completion and crashes.
3. Install any additional missing requirements, you may check [here](https://github.com/fschmid56/cpjku_dcase23) or [here](https://github.com/CPJKU/dcase2024_task1_baseline)

<h2>Prepare the Dataset</h2>

1. Ensure the h5 files for the training dataset and DIR augmentation are in the root of your working directory
2. If you do not have the h5 files, download the TAU 2022 urban acoustic scenes mobile development dataset [here](https://zenodo.org/records/6337421).
3. Extract the files to your desired dataset path
4. Ensure the meta.csv file is in .../TAU-urban-acoustic-scenes-2022-mobile-development
5. Generate h5 files for the audio data using generate_h5.py; Modify paths and filenames as desired. 
6. Generate h5 files for the dir augmentations if h5 is not provided. Modify generate_h5.py to suit the task.

<h2>Prepare the resource and metadata files:</h2>

1. Adjust the paths in datasets/dcase24_ntu_teacher.py and dcase24_student.py to point at .../TAU-urban-acoustic-scenes-2022-mobile-development (do not point at TAU-urban-acoustic-scenes-2022-mobile-development/audio)
2. In dcase24_student.py, adjust logit file paths to point at .../predictions/ensemble/yourlogit.pt. Remember to change this when performing KD with different teacher combinations

<br />
<br />

<h2>Perform Knowledge Distillation</h2>

Adjust the parameters of the baseline mode. For this demo with the 5% split:

- Set the default value of --subset to 5
- Set the default value of --resample_rate to 32000
- Set the default value of --batch_size to 256 or 128, depending on size of GPU memory
- For windows users, set the default value of num_workers to 0

run the script using command line:
```
python run_training_KD_h5.py
```
or
```
python run_training_KD_h5.py --subset 5
```

Optionally, fine-tune your own PaSST model with TAU:  <br/>

- Set the default value of --ckpt_id to None
- Set the default value of --subset between 5 to 100
- Set the default value of --n_classes to 10
- Set the default value of --resample_rate to 44100
- Set the default value of --batch_size to 256 or 128, depending on size of GPU memory
- For windows users, set the default value of num_workers to 0

run the script using command line: 
```
python run_passt_training_h5.py
```

<h2>Obtaining Teacher Logits</h2>

Reuse the same parameters you used for your teacher model and make the following changes:

- Set the default value of --ckpt_id to your model's id
- Adjust the eval_dl depending on

run the script using command line: 
```
python run_training_KD_h5.py --evaluate
```

<h2>Batching your jobs</h2>
The batch_run.py script allows you to repeatedly run the same script n number of times with the same hyperparamters. Add or edit the base_args to customize the models.

<h2>References:</h2>

[1] Koutini, K., Schlüter, J., Eghbal-Zadeh, H., & Widmer, G. (2021). Efficient training of audio transformers with patchout. arXiv preprint arXiv:2110.05069.

[2] Gemmeke, J. F., Ellis, D. P., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., ... & Ritter, M. (2017, March). Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 776-780). IEEE.

[3] T. Heittola, A. Mesarosand T. Virtanen, “TAU Urban Acoustic Scenes 2022 Mobile, Development dataset”. Zenodo, Mar. 08, 2022. doi: 10.5281/zenodo.6337421.






<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
