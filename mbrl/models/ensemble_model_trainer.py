import logging
from time import time

import numpy as np
from dotmap import DotMap
from torch.utils.data import DataLoader

from mbrl import utils
from mbrl.models.ensemblemodel import EnsembleModel
from mbrl.models.model_trainer import ModelTrainer
from mbrl.models.torchmodel import TorchModel

log = logging.getLogger(__name__)

# alonrot added:
from datetime import datetime


class EnsembleModelTrainer(ModelTrainer):
    def __init__(self, loss_cfg):
        self.loss_functions = None
        self.loss_cfg = loss_cfg

    def setup(self, model):
        assert isinstance(model, TorchModel)
        def loss_factory():
            return utils.instantiate(self.loss_cfg)

        if isinstance(model, EnsembleModel):
            ensemble_size = model.get_ensemble_size()
        else:
            ensemble_size = 1
        self.loss_functions = [loss_factory().to(device=model.get_device()) for _ in range(ensemble_size)]

    def train_model(self, model, training_dataset, testing_dataset, cfg):

        assert isinstance(model, TorchModel)
        start_time = time()
        device = model.get_device()
        # can train regular torch models too
        if isinstance(model, EnsembleModel):
            ensemble_size = model.get_ensemble_size()
            models = model.get_models()
        else:
            ensemble_size = 1
            models = [model]

        train_log = DotMap()
        train_log.time = []
        train_log.test_loss = -1
        train_log.train_loss = -1
        train_log.epochs = cfg.epochs

        # list of losses, item[i] is the last train losses for each one of the ensemble models in epoch i
        train_log.train_losses = []
        # list of losses, item[i] is the last test losses for each one of the ensemble models in epoch i
        train_log.test_losses = []

        datasets = utils.split_to_subsets(training_dataset, ensemble_size)

        if cfg.shuffle_data: # Shuffle by default
            loaders = [DataLoader(datasets[i], batch_size=cfg.batch_size, shuffle=True) for i in range(ensemble_size)]
        else: # Do not shuffle the data, if requested by the user
            loaders = [DataLoader(datasets[i], batch_size=cfg.batch_size, shuffle=False) for i in range(ensemble_size)]

        loss_functions = self.loss_functions
        optimizers = [utils.instantiate(cfg.optimizer, models[i].parameters()) for i in range(ensemble_size)]

        # alonrot added: plot training status every 20 epochs
        bunch_epoch = 20
        Nbunchep = int(cfg.epochs / bunch_epoch) + 1
        time_bunch_vec = np.zeros(Nbunchep)
        ind_c = 0

        for epoch in range(cfg.epochs):
            epoch_start_time = time()
            time_init_verbo = datetime.utcnow().timestamp()
            # maintains the last training loss in the epoch for each of the ensemble models
            ensemble_train_loss = [-1] * ensemble_size
            for batch_num, batches in enumerate(zip(*loaders)):
                for eid, batch in enumerate(batches):
                    x = batch[0].to(device=device) # Inputs
                    y = batch[1].to(device=device) # Targets

                    if x.dim() == 1:
                        x = x.unsqueeze(0).t()
                    if y.dim() == 1:
                        y = y.unsqueeze(0).t()

                    optimizers[eid].zero_grad()
                    py = models[eid].forward(x) # x: NN input; py: NN output; y: target
                    # This can potentially be slow, consider enabling selectively.
                    utils.assert_no_nans(py)
                    loss = loss_functions[eid](py, y)
                    loss.backward()
                    optimizers[eid].step()
                    ensemble_train_loss[eid] = loss.item()

            # alonrot added:
            if epoch % bunch_epoch == 0:
                log.info("\nTraining, epoch: {0:d} / {1:d}".format(epoch,cfg.epochs))
                time_bunch_vec[ind_c] = datetime.utcnow().timestamp() - time_init_verbo
                log.info("Elapsed time: {0:2.2f} [sec]".format(time_bunch_vec[ind_c]))
                time_bunch_avg = np.mean(time_bunch_vec[0:ind_c+1])
                log.info("Average time per {0:d} epochs: {1:2.2f} [sec]".format(bunch_epoch,time_bunch_avg))
                log.info("Remaining: {0:2.2f} [sec]".format( (Nbunchep-(ind_c-1))*time_bunch_avg ))
                ind_c += 1

            train_log.train_loss = np.mean(ensemble_train_loss)
            train_log.train_losses.append(ensemble_train_loss)

            if testing_dataset is not None:
                testing_loader = DataLoader(testing_dataset, batch_size=cfg.batch_size, shuffle=False)
                ensemble_test_loss = [-1] * ensemble_size
                for _, data in enumerate(testing_loader):
                    inputs = batch[0].to(device=device)
                    targets = batch[1].to(device=device)
                    for eid, model in enumerate(models):
                        outputs = model(inputs)
                        loss = loss_functions[eid](outputs, targets)
                        ensemble_test_loss[eid] = loss.item()
                train_log.test_loss = np.mean(ensemble_test_loss)
                train_log.test_losses.append(ensemble_test_loss)

            epoch_time = time() - epoch_start_time
            train_log.time.append(epoch_time)
            log.debug(
                f"#E{epoch}, Train loss={train_log.train_loss:.4f}, test loss={train_log.test_loss:.4f} ({epoch_time:.2f}s)")

        train_log.total_time = time() - start_time

        return train_log
