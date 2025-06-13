# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import atexit
import logging as base_logging
from nemo.utils import logging

logging.setLevel(logging.ERROR)

import torch
import torch.multiprocessing as mp
from custom_callbacks import CustomCallback, CustomMegatronGPTSFTModel, MetricsLogger
from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from omegaconf.omegaconf import OmegaConf
from lightning_fabric.utilities.distributed import _destroy_dist_connection


class IgnoreRegexPatternFilter(base_logging.Filter):
    def __init__(self, patterns):
        self.patterns = patterns

    def filter(self, record):
        message = record.getMessage()
        return not any(pattern in message for pattern in self.patterns)

base_logging.getLogger().addFilter(IgnoreRegexPatternFilter(["running build_ext"]))
base_logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
base_logging.getLogger("DotProductAttention").setLevel(base_logging.WARNING)


def get_rank():
    return int(os.getenv("SLURM_PROCID", 0))

def string_to_bool(text):
    if text is None or text.lower() == "false":
        return False
    elif text.lower() == "true":
        return True
    raise ValueError("The string must be 'true' or 'false', case insensitive.")


mp.set_start_method("spawn", force=True)

OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("floor_div", lambda x, y: x // y)
OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)

load_llama_ckpt = string_to_bool(os.getenv("LOAD_CKPT", "True"))


def create_model(cfg, trainer):
    peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]
    path = cfg.model.restore_from_path if load_llama_ckpt else "/workspace/ft-llm/conf/base"
    model_cfg = CustomMegatronGPTSFTModel.merge_cfg_with(path, cfg)
    if load_llama_ckpt:
        model = CustomMegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
    else:
        model = CustomMegatronGPTSFTModel(model_cfg, trainer)
    model.add_adapter(peft_cfg_cls(model_cfg))
    return model


@hydra_runner(config_path="conf", config_name="megatron_gpt_peft_tuning_config")
def main(cfg) -> None:
    OmegaConf.resolve(cfg)
    if get_rank() == 0:
        print("\n\n************** Experiment configuration ***********")
        print(f"\n{OmegaConf.to_yaml(cfg)}")
    mllogger = MLLoggerWrapper(PyTCommunicationHandler())

    custom_callback = CustomCallback(cfg, mllogger)
    precision = cfg.trainer.precision
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer(callbacks=[custom_callback])
    cfg.trainer.precision = precision
    trainer.loggers.append(MetricsLogger(cfg, mllogger, trainer))
    model = create_model(cfg, trainer)
    gc.disable()
    s = torch.cuda.Stream()
    torch.cuda.set_stream(s)
    trainer.fit(model)
    atexit.unregister(_destroy_dist_connection)


if __name__ == "__main__":
    main()
