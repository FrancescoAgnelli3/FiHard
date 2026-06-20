import os
import socket
import math
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

OmegaConf.register_new_resolver("eval", eval)

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.handlers import Checkpoint, DiskSaver

from src.train_utils import setup_validation_autoencoder, create_train_dataloaders, resume_training, create_model
from src.data import create_skeleton
from src.utils.reproducibility import set_seed
from src.utils.tensorboard import setup_tb_logging_valid, setup_tb_logging_train
from src.core.trainer import AutoEncoderTrainer, setup_scheduler_step
from src.utils.config import flat_hydra_config, config_exp_folder



os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def train(output_log_path: str,
          num_epochs: int,
          save_frequency: int = 100,
          eval_frequency: int = None,
          num_iteration_eval: int = 0,
          device: str = 'cpu',
          seed: int = 52345,
          num_iter_perepoch: int = 580,
          detect_anomaly: bool = False,
          if_resume_training: bool = False, 
          if_run_validation: bool = False,
          **kwargs) -> None:
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # Init seed
    set_seed(seed)
    torch.set_default_dtype(torch.float64 if kwargs["dtype"]== "float64" else torch.float32)
    print(f"Default datatype: {torch.get_default_dtype()} . Using device: {device} - {torch.cuda.get_device_name(0).replace(' ', '_') if torch.cuda.is_available() else ''}")

    skeleton = create_skeleton(**kwargs)

    # # Create model
    model = create_model(device, skeleton=skeleton, num_nodes=skeleton.num_nodes,
                   node_types=skeleton.nodes_type_id, **kwargs['autoenc_arch'], **kwargs)

    print(f"Created Autoencoder with {str(round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000, 3))}M trainable parameters.")

    # Load dataset
    data_loader_train, data_loader_eval, data_loader_train_eval, random_state_manager = create_train_dataloaders(skeleton=skeleton, if_run_validation=if_run_validation, if_resume_training=if_resume_training, **kwargs)
   
    model_trainer = AutoEncoderTrainer(model, iter_per_epoch=num_iter_perepoch, **kwargs)
    es_enabled = bool(kwargs.get("early_stopping_enabled", False))
    es_patience = max(1, int(kwargs.get("early_stopping_patience", 20)))
    es_min_delta = float(kwargs.get("early_stopping_min_delta", 1e-4))
    es_warmup = max(0, int(kwargs.get("early_stopping_warmup", 0)))
    es_monitor = str(kwargs.get("early_stopping_monitor", "train_loss")).strip().lower()
    es_best = None
    es_bad_epochs = 0
    last_val_loss = None

    
    def set_epoch_seed(engine: Engine):
        set_seed(seed + engine.state.epoch)
        random_state_manager.reseed_generator(seed + engine.state.epoch)
        
    def preprocess(engine: Engine):
        engine.state.batch =  [t.to(device) for t in engine.state.batch[:2]]
        
    # Define Training Engines
    trainer = Engine(model_trainer.train_step)
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'Loss')
    trainer.add_event_handler(Events.ITERATION_STARTED, preprocess)
    trainer.add_event_handler(Events.EPOCH_STARTED, set_epoch_seed)
    # Set up schedulers
    setup_scheduler_step(model_trainer, train_engine=trainer)
    if model_trainer.curriculum_scheduler is not None:
        trainer.add_event_handler(Events.ITERATION_STARTED, model_trainer.curriculum_scheduler)

    pbar = ProgressBar()
    pbar.attach(trainer, ['Loss'])

    # Setup tensorboard logging and progressbar
    tb_logger = TensorboardLogger(log_dir=os.path.join(output_log_path, 'tb'))
    setup_tb_logging_train(tb_logger, trainer=trainer, optimizer=model_trainer.optimizer, 
                        model=model, train_out_transform=lambda out: {"loss": out[0],  "pred horiz": out[-1]})
    if if_run_validation:
        evaluator, evaluator_train = setup_validation_autoencoder(model, preprocess, model_trainer, skeleton)
        pbar.attach(evaluator)
        pbar.attach(evaluator_train)
        setup_tb_logging_valid(tb_logger, trainer=trainer, evaluator=evaluator, evaluator_train=evaluator_train, 
                                loss_names=["Loss"], metric_names=['ADE', 'FDE',  "MPJPE_AVG",  "MPJPE"])

    # Define handlers for saving checkpoints
    objects_to_checkpoint = {'trainer': trainer, 'model': model,"random_states": random_state_manager, **model_trainer.objects_to_checkpoint()}
    checkpoint_handler = Checkpoint(objects_to_checkpoint, DiskSaver(os.path.join(output_log_path, "checkpoints"), create_dir=True, require_empty=False), n_saved=20, 
                                    global_step_transform= lambda engine, event_name: trainer.state.epoch, score_function=lambda engine: -1* round(engine.state.metrics['MPJPE_AVG'].item(), 7), score_name="val_mpjpe")
    static_checkpoint_handler = Checkpoint(objects_to_checkpoint, checkpoint_handler.save_handler, n_saved=None, global_step_transform= lambda engine, event_name: trainer.state.epoch)
    if if_resume_training:
        checkpoint = torch.load(kwargs['load_path'])
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)  
        print("Resuming training from: epoch = ", trainer.state.epoch, '. Iteration = ', trainer.state.iteration)
    
    
    #Regularly save checkpoints during training
    @trainer.on(Events.EPOCH_COMPLETED(every=save_frequency))
    def save_latest_model(engine: Engine):
        static_checkpoint_handler(trainer)
    
    #Setup evaluation process between epochs
    if if_run_validation:
        @trainer.on(Events.EPOCH_COMPLETED(every=eval_frequency))
        def train_epoch_completed(engine: Engine):
            nonlocal last_val_loss
            set_seed(0)
            random_state_manager.reseed_generator(0)
            evaluator.run(data_loader_eval)
            evaluator_train.run(data_loader_train_eval, epoch_length=num_iteration_eval)
            val_metric = evaluator.state.metrics.get("Loss")
            if val_metric is not None:
                try:
                    last_val_loss = float(val_metric)
                except (TypeError, ValueError):
                    last_val_loss = None

            checkpoint_handler(evaluator)
            set_seed(seed + trainer.state.epoch)
            random_state_manager.reseed_generator(seed + trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def maybe_early_stop(engine: Engine):
        nonlocal es_best, es_bad_epochs
        if not es_enabled:
            return
        epoch_idx = int(engine.state.epoch)
        if epoch_idx <= es_warmup:
            return
        train_metric = engine.state.metrics.get("Loss")
        train_loss = None
        if train_metric is not None:
            try:
                train_loss = float(train_metric)
            except (TypeError, ValueError):
                train_loss = None
        if es_monitor == "val_loss":
            monitored = last_val_loss
        elif es_monitor in {"train_loss", "loss"}:
            monitored = train_loss
        else:
            monitored = last_val_loss if last_val_loss is not None else train_loss
        if monitored is None or not math.isfinite(float(monitored)):
            return
        monitored_f = float(monitored)
        improved = es_best is None or monitored_f < (float(es_best) - es_min_delta)
        if improved:
            es_best = monitored_f
            es_bad_epochs = 0
            return
        es_bad_epochs += 1
        if es_bad_epochs >= es_patience:
            print(
                f"[SkeletonDiffusion][Autoencoder][EarlyStop] epoch={epoch_idx} "
                f"best={float(es_best):.6f} current={monitored_f:.6f}"
            )
            engine.terminate()


    trainer.run(data_loader_train, epoch_length=num_iter_perepoch, max_epochs=num_epochs) 
    static_checkpoint_handler(trainer)  # Save the last model after training
    tb_logger.close()



@hydra.main(config_path="./configs/config_train_autoencoder", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # for backwards compatibility with the code
    cfg = flat_hydra_config(cfg)

    assert cfg['num_workers'] >=1, "Necessary for reproducibility issues"
    
    
    if cfg['if_resume_training']:
        print("Configuration file will be ignored. Considering old configuration file.")
        cfg = resume_training(cfg)
    else: 
        # Start from scratch
        cfg = config_exp_folder(cfg)
    print("Traing data saved at ", cfg['output_log_path'])
    print("Training on device: ", f"{socket.gethostname().split('.')[0]}")
    train(**cfg)
    



if __name__ == '__main__':
    main()


    
