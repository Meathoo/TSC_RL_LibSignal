import logging
import json
import os
from datetime import datetime
from common.registry import Registry


@Registry.register_task('base')
class BaseTask:
    '''
    Register BaseTask, currently support TSC task.
    '''
    def __init__(self, trainer):
        self.trainer = trainer

    def run(self):
        raise NotImplementedError

    def _process_error(self, e):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
        ):
            for name, parameter in self.trainer.agents.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )


@Registry.register_task("tsc")
class TSCTask(BaseTask):
    '''
    Register Traffic Signal Control task.
    '''
    def _save_training_hyperparameters(self):
        output_dir = Registry.mapping['logger_mapping']['path'].path
        os.makedirs(output_dir, exist_ok=True)

        hyperparameters = {
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'command': Registry.mapping['command_mapping']['setting'].param,
            'trainer': Registry.mapping['trainer_mapping']['setting'].param,
            'model': Registry.mapping['model_mapping']['setting'].param,
            'world': Registry.mapping['world_mapping']['setting'].param,
            'logger': Registry.mapping['logger_mapping']['setting'].param,
        }

        output_file = os.path.join(output_dir, 'hyperparameters.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(hyperparameters, f, ensure_ascii=False, indent=2, sort_keys=True)

        logging.info(f"Hyperparameters saved to: {output_file}")

    def run(self):
        '''
        run
        Run the whole task, including training and testing.

        :param: None
        :return: None
        '''
        try:
            if Registry.mapping['model_mapping']['setting'].param['train_model']:
                self.trainer.train()
                self._save_training_hyperparameters()
            if Registry.mapping['model_mapping']['setting'].param['test_model']:
                self.trainer.test()
        except RuntimeError as e:
            self._process_error(e)
            raise e
