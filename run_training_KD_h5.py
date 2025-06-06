import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import torchaudio
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import transformers
import wandb
import json
import torch.nn as nn
import os

from helpers.lr_schedule import exp_warmup_linear_down
from datasets.dcase24_ntu_student import ntu_get_training_set_dir, ntu_get_test_set, ntu_get_eval_set, open_h5, close_h5 # for raw wav get_training_set, get_test_set, get_eval_set
from helpers.init import worker_init_fn
from models.baseline import get_model
from helpers.utils import mixstyle
from helpers import nessi

torch.set_float32_matmul_precision("high")

## In FocusNet, we need a baseline or it's logits to adjust the weighting of the loss for student logits.
## We load logits from a .pt file by calling logits = torch.load(logits).float()


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse, contains all configurations for our experiment

        # module for resampling waveforms on the fly
        resample = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.sample_rate
        )

        # module to preprocess waveforms into log mel spectrograms
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.window_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max
        )

        freqm = torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True)
        timem = torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)

        self.mel = torch.nn.Sequential(
            resample,
            mel
        )

        self.mel_augment = torch.nn.Sequential(
            freqm,
            timem
        )
        
        # the baseline model
        self.model = get_model(n_classes=config.n_classes,
                               in_channels=config.in_channels,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier,
                               expansion_rate=config.expansion_rate
                               )
        self.kl_div_loss = nn.KLDivLoss(log_target=True, reduction="none") # KL Divergence loss for soft, check log_target 
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        # pl 2 containers:
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.logit_stand = config.logit_stand
        
    def normalize(pl,logit):
        # print(f"logit before standardization: {logit}")
        mean = logit.mean(dim=-1, keepdims=True)
        # print(f"mean of logits, dimensions preserved for logit-wise deduction: {mean}")
        stdv = logit.std(dim=-1, keepdims=True)
        logit = logit-mean/ (1e-7+stdv)
        # print(f"logit after standardization: {logit}")
        return logit
    
    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # schedule_lambda = \
        #     exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
        #                            self.config.last_lr_value)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        
        #For regular training
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': lr_scheduler
        # }

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, files, labels, devices, cities, teacher_logits = train_batch
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device=x.device)
        if self.config.mixstyle_p > 0:
            # frequency mixstyle
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
        y_hat = self.model(x.cuda()) # This is the outputs
        # At this point we want to perform KLdiv loss      
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        label_loss = samples_loss.mean()
        # Temperature adjusted probabilities of teacher and student
        if self.logit_stand:
            y_hat_stand = self.normalize(y_hat)
            with torch.cuda.amp.autocast():                
                y_hat_soft = F.log_softmax(y_hat_stand / self.config.temperature, dim=-1)
                teacher_logits = F.log_softmax(teacher_logits / self.config.temperature, dim=-1)
        else:
            with torch.cuda.amp.autocast():
                y_hat_soft = F.log_softmax(y_hat / self.config.temperature, dim=-1)
                teacher_logits = F.log_softmax(teacher_logits / self.config.temperature, dim=-1)
        kd_loss = self.kl_div_loss(y_hat_soft, teacher_logits).mean() #
        kd_loss = kd_loss * (self.config.temperature ** 2)
        loss = self.config.kd_lambda * label_loss + (1 - self.config.kd_lambda) * kd_loss
        # loss = kd_loss
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss)
        results = {"loss": loss, "label_loss": label_loss * self.config.kd_lambda,
                   "kd_loss": kd_loss * (1 - self.config.kd_lambda)}

        return results
        # return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device=x.device)
        y_hat = self.forward(x.cuda())
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        '''# log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        results = {k: v.detach() for k, v in results.items()}'''
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        '''# log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'val' for logging'''
        self.log_dict({"val/" + k: logs[k] for k in logs})
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, cities = test_batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device=x.device)
        # maximum memory allowance for parameters: 128 KB
        # baseline has 61148 parameters -> we can afford 16-bit precision
        # since 61148 * 16 bit ~ 122 kB

        # assure fp16
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x.cuda())
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'test' for logging
        self.log_dict({"test/" + k: logs[k] for k in logs})
        self.test_step_outputs.clear()

    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch

        # assure fp16
        self.model.half()

        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)

        return files, y_hat


def train(config):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'24 Task 1.",
        tags=["DCASE24"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )

    # train dataloader
    assert config.subset in {100, 50, 25, 10, 5}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."
    
    # get pointer to h5 file containing audio samples
    hf_in = open_h5('h5py_audio_wav')
    hmic_in = open_h5('h5py_mic_wav_1')

    # get_training set already as logic to handle dir_prob=0
    train_dl = DataLoader(dataset=ntu_get_training_set_dir(config.subset, config.dir_prob, hf_in, hmic_in),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          
                          shuffle=True)
    
    test_dl = DataLoader(dataset=ntu_get_test_set(hf_in),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)
    
    
    if config.logit_stand:
        print(f"Logit Standardization in effect")
    else:
        print(f"No Logit Standardization")
    # create pytorch lightening module
    ckpt_id = None if config.ckpt_id == "None" else config.ckpt_id
    if ckpt_id is not None:
        ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
        assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
        for file in os.listdir(ckpt_dir):
            if "epoch" in file:
                ckpt_file = os.path.join(ckpt_dir,file) # choosing the best model ckpt
                print(f"found ckpt file: {file}")
        pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    else:
        pl_module = PLModule(config) # this initializes the model pre-trained on audioset
    # get model complexity from nessi and log results to wandb
    sample = next(iter(test_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    # log MACs and number of parameters for our model
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='gpu',
                         devices=eval(config.gpu),
                         num_sanity_val_steps=0,
                         precision=config.precision, detect_anomaly=True,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_last=True, monitor = "val/loss",save_top_k=1),]
                         )
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, test_dl)

    # final test step
    # here: use the validation split
    trainer.test(ckpt_path='best', dataloaders=test_dl)

    
    # close file pointer to h5 file 
    close_h5(hf_in)
    close_h5(hmic_in)

    wandb.finish()


def evaluate(config):
    import os
    from sklearn import preprocessing
    import pandas as pd
    import torch.nn.functional as F
    from datasets.dcase24 import dataset_config

    assert config.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    # ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    for file in os.listdir(ckpt_dir):
        if "epoch" in file:
            ckpt_file = os.path.join(ckpt_dir,file) # choosing the best model ckpt
    # ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select" \
                                      f"the desired checkpoint."

    # create folder to store predictions
    os.makedirs("predictions", exist_ok=True)
    out_dir = os.path.join("predictions", config.ckpt_id)
    os.makedirs(out_dir, exist_ok=True)

    
    # Open h5 file once       
    hf_in = open_h5('h5py_audio_wav')
    eval_h5 = open_h5('eval_h5py') # generate your own eval_h5 during the DCASE challenge.
    # load lightning module from checkpoint

    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    trainer = pl.Trainer(logger=False,
                         accelerator='gpu',
                         devices=[0],
                         precision=config.precision)
    
    # evaluate lightning module on development-test split
    test_dl = DataLoader(dataset=ntu_get_test_set(hf_in),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size,
                         pin_memory=True)
    
        # get model complexity from nessi
    sample = next(iter(test_dl))[0][0].unsqueeze(0).to(pl_module.device)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)

    print(f"Model Complexity: MACs: {macs}, Params: {params}")
    assert macs <= nessi.MAX_MACS, "The model exceeds the MACs limit and must not be submitted to the challenge!"
    assert params <= nessi.MAX_PARAMS_MEMORY, \
        "The model exceeds the parameter limit and must not be submitted to the challenge!"

    allowed_precision = int(nessi.MAX_PARAMS_MEMORY / params * 8)
    print(f"ATTENTION: According to the number of model parameters and the memory limits that apply in the challenge,"
          f" you are allowed to use at max the following precision for model parameters: {allowed_precision} bit.")

    # obtain and store details on model for reporting in the technical report
    info = {}
    info['MACs'] = macs
    info['Params'] = params
    res = trainer.test(pl_module, test_dl)
    info['test'] = res

    
    
        
    # eval_dl = DataLoader(dataset=ntu_get_eval_set(hf_in), # used to generate teacher logits if desired. 
    #                      worker_init_fn=worker_init_fn,
    #                      num_workers=config.num_workers,
    #                      batch_size=config.batch_size)
                         
    eval_dl = DataLoader(dataset=ntu_get_eval_set(eval_h5), # Generate predictions on evaluation set. 
                         worker_init_fn=worker_init_fn,     # Remember to adjust your csv path in dcase24_ntu_student.py
                         num_workers=config.num_workers,    
                         batch_size=config.batch_size)

    # # generate predictions on evaluation set              # Used for wav files
    # eval_dl = DataLoader(dataset=get_eval_set(),          # Remember to adjust your csv path in dcase24_ntu_student.py
    #                      worker_init_fn=worker_init_fn,
    #                      num_workers=config.num_workers,
    #                      batch_size=config.batch_size)

    predictions = trainer.predict(pl_module, dataloaders=eval_dl,ckpt_path=ckpt_file)
    # all filenames
    all_files = [item[len("audio/"):] for files, _ in predictions for item in files]
    # all predictions
    logits = torch.cat([torch.as_tensor(p) for _, p in predictions], 0)
    all_predictions = F.softmax(logits.float(), dim=1)

    # write eval set predictions to csv file
    df = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    le = preprocessing.LabelEncoder()
    le.fit_transform(df[['scene_label']].values.reshape(-1))
    class_names = le.classes_
    df = {'filename': all_files}
    scene_labels = [class_names[i] for i in torch.argmax(all_predictions, dim=1)]
    df['scene_label'] = scene_labels
    for i, label in enumerate(class_names):
        df[label] = logits[:, i]
    df = pd.DataFrame(df)

    # save eval set predictions, model state_dict and info to output folder
    df.to_csv(os.path.join(out_dir, 'output.csv'), sep='\t', index=False)
    torch.save(pl_module.model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
    with open(os.path.join(out_dir, "info.json"), "w") as json_file:
        json.dump(info, json_file)

    
    close_h5(hf_in)
    close_h5(eval_h5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 24 argument parser')

    # general
    parser.add_argument('--project_name', type=str, default="NTU_ASC_Baseline")
    parser.add_argument('--experiment_name', type=str, default="NTU_KD_32K_FMS_DIR")
    parser.add_argument('--num_workers', type=int, default=0)  # number of workers for dataloaders
    parser.add_argument('--precision', type=str, default="32")
    parser.add_argument('--gpu',type=str,default="[0]")
    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, required=False, default=None)  # for loading trained model, corresponds to wandb id k0r7gu0i

    # dataset
    # subset in {100, 50, 25, 10, 5}
    parser.add_argument('--orig_sample_rate', type=int, default=44100) # do not modify
    parser.add_argument('--subset', type=int, default=5)

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network (3 main dimensions to scale the baseline)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channels_multiplier', type=float, default=1.8)
    parser.add_argument('--expansion_rate', type=float, default=2.1)

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    # parser.add_argument('--mixstyle_p', type=float, default=0)
    parser.add_argument('--mixstyle_p', type=float, default=0.4)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # parser.add_argument('--roll_sec', type=int, default=0)
    parser.add_argument('--roll_sec', type=int, default=0)  # roll waveform over time
    parser.add_argument('--dir_prob', type=float, default=0.6)  # prob. to apply device impulse response augmentation default = 0.6
    
    ## knowledge distillation
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--kd_lambda', type=float, default=0.02) # default is 0.02
    parser.add_argument('--logit_stand', action='store_true', default=False) # Implements logit standardisation
    
    # # peak learning rate (in cosine schedule)
    parser.add_argument('--lr', type=float, default=0.005) # 
    parser.add_argument('--warmup_steps', type=int, default=100) # default = 2000 for the 100% split
    
    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)   # Required to keep within challenge requirements, unless changes to fft are made
    parser.add_argument('--window_length', type=int, default=3072)  # in samples (corresponds to 96 ms)
    parser.add_argument('--hop_length', type=int, default=500)  # in samples (corresponds to ~16 ms)
    parser.add_argument('--n_fft', type=int, default=4096)  # length (points) of fft, e.g. 4096 point FFT
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    # parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--f_min', type=int, default=0)  # mel bins are created for freqs. between 'f_min' and 'f_max'
    parser.add_argument('--f_max', type=int, default=None)

    args = parser.parse_args()
    if args.evaluate:
        evaluate(args)
    else:
        train(args)
