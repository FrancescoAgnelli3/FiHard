import torch
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from typing import Sequence

from torch.utils.data import DataLoader
from ignite.handlers import Checkpoint
import os

from .utils.config import init_obj
from .utils.reproducibility import seed_worker, seed_eval_worker, RandomStateDict
import src.data.loaders as dataset_type
from .utils.load import get_latest_model_path
from .inference_utils import create_model
from .config_metrics import MeanPerJointPositionError, FinalDisplacementError, ade, MetricStorer
from .config_metrics import apd, ade, limb_length_variance, MetricStorer, MeanPerJointPositionError, choose_best_sample

def create_train_dataloaders(batch_size, batch_size_eval, num_workers, skeleton, if_run_validation=True, if_resume_training=False, **config):
    
    random_state_manager = RandomStateDict()
    if if_resume_training:
        checkpoint = torch.load(config['load_path'])
        Checkpoint.load_objects(to_load={"random_states": random_state_manager}, checkpoint=checkpoint)  
    dataset_train = init_obj(config, 'dataset_type', dataset_type, split="train", skeleton=skeleton, **(config['data_loader_train']))
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, worker_init_fn = seed_worker, pin_memory= True,
                                num_workers=num_workers, generator=random_state_manager.generator)  
     
    if if_run_validation:
        dataset_eval = init_obj(config, 'dataset_type', dataset_type, split="valid", skeleton=skeleton, **(config['data_loader_valid']))
        dataset_eval_train = init_obj(config, 'dataset_type', dataset_type, split="train", skeleton=skeleton, **(config['data_loader_train_eval']))
        data_loader_eval = DataLoader(dataset_eval, shuffle=False, worker_init_fn=seed_eval_worker, batch_size=batch_size_eval, num_workers=1, pin_memory= True)    
        data_loader_train_eval = DataLoader(dataset_eval_train, shuffle=False, worker_init_fn=seed_eval_worker, batch_size=batch_size_eval, num_workers=1, pin_memory= True)
    else: 
        data_loader_eval = None
        data_loader_train_eval = None
    return data_loader_train, data_loader_eval, data_loader_train_eval, random_state_manager

def resume_training(cfg):
    #output folder has been already created.
    assert 'output_log_path' in cfg

    # decide whether to start from scratch (default if no checkpoints), from latest save (if checkpoints exists), or from given path (if load_path is given)

    assert len(os.listdir(os.path.join(cfg['output_log_path'], 'checkpoints'))) != 0, "Checkpoints folder is empty. Please provide a valid path to load from."
        
    if len(cfg['load_path']) == 0:
        #  load latest model
        cfg['load_path'] = get_latest_model_path(os.path.join(cfg['output_log_path'], 'checkpoints'))
        print("Loading latest epoch: ", cfg['load_path'].split('/')[-1])
    else: 
        output_path = os.path.dirname(os.path.dirname(cfg['load_path']))
        assert cfg['output_log_path'] ==  output_path
        cfg['output_log_path'] = output_path
    return cfg

def setup_validation_autoencoder(model, preprocess, model_trainer, skeleton):
    def validation_step_autoencoder(engine: Engine, batch: Sequence[torch.Tensor]): 
        with torch.no_grad():
            model_out, y, x, z = model_trainer.validation_step(engine, batch)
            model_out_metric = skeleton.transform_to_metric_space(model_out)
            y_metric = skeleton.transform_to_metric_space(y)
            return (model_out_metric, y_metric), (model_out, y), z, 

    # Define ignite metrics
    ade_metric = MetricStorer(output_transform=lambda x: ade(pred=x[0][0].unsqueeze(1), target=x[0][1])) 
    mpjpe = MeanPerJointPositionError(output_transform=lambda x: (x[0]))
    fde = FinalDisplacementError(output_transform=lambda x: (x[0]))
    loss_metric = Loss(model.loss, output_transform=lambda x: (x[1]))
    
    def add_avg_metrics_to_existing_metrics(engine: Engine):
        for metric in list(engine.state.metrics.keys()):
            if metric in ["MPJPE"]:
                engine.state.metrics[metric+"_AVG"] =  engine.state.metrics[metric].mean(0)

    def attach_losses_to_engine(engine):
        engine.add_event_handler(Events.ITERATION_STARTED, preprocess)
        loss_metric.attach(engine, 'Loss')

    def attach_evalmetrics_to_engine(engine):
        mpjpe.attach(engine, 'MPJPE')
        ade_metric.attach(engine, 'ADE')
        fde.attach(engine, 'FDE')
        engine.add_event_handler(Events.COMPLETED, add_avg_metrics_to_existing_metrics)

    # Define Evaluation Engines and attach metrics
    evaluator = Engine(validation_step_autoencoder)
    attach_losses_to_engine(evaluator)
    attach_evalmetrics_to_engine(evaluator)

    # Evaluate on Train dataset
    evaluator_train = Engine(validation_step_autoencoder)
    attach_losses_to_engine(evaluator_train)
    attach_evalmetrics_to_engine(evaluator_train)
    
    return evaluator, evaluator_train

def setup_validation_diffusion(model, preprocess, diffusion_trainer, skeleton):
    def validation_step(engine: Engine, batch: Sequence[torch.Tensor]): 
        model_out, y, _, x = diffusion_trainer.validation_step(engine=engine, batch=batch)
        model_out_metric = skeleton.transform_to_metric_space(model_out)
        y_metric = skeleton.transform_to_metric_space(y)
        return {'pred':model_out_metric, 'target':y_metric}, (model_out, y), x
   
    # Define ignite metrics
    mpjpe = MeanPerJointPositionError(output_transform=lambda x: choose_best_sample( x[0]['pred'], x[0]['target'],))
    apd_metric = MetricStorer(output_transform=lambda x: apd(**x[0]))
    ade_metric = MetricStorer(output_transform=lambda x: ade(**x[0]))
    limbl_variance = MetricStorer(output_transform=lambda x: limb_length_variance(pred=x[0]['pred'], limbseq=skeleton.get_limbseq(), mode='mean'))
    loss_metric = Loss(model.loss, output_transform=lambda x: choose_best_sample(*x[-2]))
        
    def add_avg_metrics_to_existing_metrics(engine: Engine):
        for metric in list(engine.state.metrics.keys()):
            if metric in ["MPJPE"]:
                engine.state.metrics[metric+"_AVG"] =  engine.state.metrics[metric].mean(0)
        engine.state.metrics["LearningRate"] = diffusion_trainer.opt.param_groups[0]['lr']
         
    def attach_evalmetrics_to_engine(engine):
        engine.add_event_handler(Events.ITERATION_STARTED, preprocess)
        mpjpe.attach(engine, 'MPJPE')
        apd_metric.attach(engine, 'APD')
        ade_metric.attach(engine, 'ADE')
        limbl_variance.attach(engine, 'LLVar')
        engine.add_event_handler(Events.COMPLETED, add_avg_metrics_to_existing_metrics)

    def attach_losses_to_engine(engine):
        loss_metric.attach(engine, 'Loss')

    # Define Evaluation and Test Engines and attach metrics
    evaluator = Engine(validation_step)
    attach_evalmetrics_to_engine(evaluator)
    attach_losses_to_engine(evaluator)

    evaluator_train = Engine(validation_step)
    attach_evalmetrics_to_engine(evaluator_train)
    attach_losses_to_engine(evaluator_train)
   
    return evaluator, evaluator_train
