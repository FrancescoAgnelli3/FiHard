import torch
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, WeightsScalarHandler, WeightsHistHandler, GradsHistHandler, GradsScalarHandler
from ignite.engine import Engine, Events
from torch.optim import Optimizer



def setup_tb_logging_valid(tb_logger: TensorboardLogger, trainer: Engine, evaluator: Engine, evaluator_train: Engine,
                  loss_names=["Loss"], metric_names=[ "ADE", "FDE", "MPJPE_AVG", "MPJPE"]):
    tb_custom_scalar_layout = {}
    #　We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
    # `trainer` instead of `evaluator`.
    tb_logger.attach_output_handler(
        evaluator_train,
        event_name=Events.COMPLETED,
        **{'tag': "losses/train",
        'metric_names':   loss_names,
        'global_step_transform': lambda engine, event_name: trainer.state.epoch 
        }
    )
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.COMPLETED,
        **{'tag': "losses/validation",
        'metric_names': loss_names,
        'global_step_transform': lambda engine, event_name: trainer.state.epoch
        }
    )
        
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.COMPLETED,
        **{'tag': "validation",
        'metric_names': loss_names + metric_names,
        'global_step_transform': lambda engine, event_name: trainer.state.epoch 
        }
    )

    tb_logger.attach_output_handler(
        evaluator_train,
        event_name=Events.COMPLETED,
        **{'tag': "validation_train",
        'metric_names': loss_names + metric_names,
        'global_step_transform': lambda engine, event_name: trainer.state.epoch 
        }
    )

    tb_custom_scalar_layout = {
        **tb_custom_scalar_layout,
        **{
            'Validation Metrics': {
                'MPJPE_T': ['Multiline', [rf"validation_train/MPJPE_T/.*"]],
            }
        }
    }
    tb_logger.writer.add_custom_scalars(tb_custom_scalar_layout)

def setup_tb_logging_train(tb_logger: TensorboardLogger, trainer: Engine, optimizer: Optimizer,
                                 model: torch.nn.Module, train_out_transform: callable):
    """
    Logs standard information to tensorboard during training.
    """
    # Attach the logger to the trainer to log training loss at each iteration
    
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        **{
            'tag': 'training',
            'output_transform': train_out_transform,
        }
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        **{
            'optimizer': optimizer,
            'param_name': 'lr',
            'tag': 'training'
        }
    )
    
    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.EPOCH_STARTED,
        **{
            'optimizer': optimizer,
            'param_name': 'lr',
            'tag': 'losses'
        }
    )

    # Attach the logger to the trainer to log model's weights norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        log_handler=WeightsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        log_handler=GradsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=GradsHistHandler(model)
    )
