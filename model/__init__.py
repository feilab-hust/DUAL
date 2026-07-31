import logging
from model.model_stage1 import DUALStage1
from model.model_stage2 import DUALStage2
def create_model(opt, stage, custom_s1=None):
    if stage == 1:
        m = DUALStage1(opt, custom_s1=custom_s1)
    elif stage == 2:
        m = DUALStage2(opt, custom_s1=custom_s1)
    else:
        raise ValueError(f"Invalid stage: {stage}. Expected 1 or 2 ")

    logger = logging.getLogger(opt['phase'])
    logger.info('[{:s}] model is created.'.format(m.__class__.__name__))
    return m
