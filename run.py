
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger  # 추가된 부분

import os
import copy

from config import ex
from model.face_tts import FaceTTS
from model.myface_tts import MyFaceTTS
from data import _datamodules

@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = _datamodules["dataset_" + _config["dataset"]](_config)
    
    local_checkpoint_dir="/mnt/bear2/users/jungji/facetts/logs"
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    logger = TensorBoardLogger(local_checkpoint_dir, name="my_model")

    checkpoint_callback_epoch = pl.callbacks.ModelCheckpoint(
        dirpath=f"{local_checkpoint_dir}/{logger.version}",
        save_top_k=1,
        verbose=True,
        monitor="val/total_loss",
        mode="min",
        save_last=True,
        filename='{epoch}-{step}-last',
        every_n_train_steps=50,
        auto_insert_metric_name=True,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # model = FaceTTS(_config)
    model = MyFaceTTS(_config,teacher=True)

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [checkpoint_callback_epoch, lr_callback, model_summary_callback]
    gpus=[0, 1]
    num_gpus = len(gpus)

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    
    trainer = pl.Trainer(
        gpus=gpus,
        num_nodes=_config["num_nodes"],
        strategy=DDPPlugin(gradient_as_bucket_view=True, find_unused_parameters=True),
        max_steps=max_steps,
        callbacks=callbacks,
        accumulate_grad_batches=2,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        weights_summary="top",
        logger=logger,
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
    else:
        trainer.test(model, datamodule=dm, ckpt_path=_config["resume_from"])
