import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
import logging
import copy
import logging

import numpy as np
from torch.distributions.laplace import Laplace
import math
from .dp_sgd.utils import get_data_loader, get_sigma, restore_param, checkpoint, adjust_learning_rate, process_grad_batch

from backpack import backpack, extend
from backpack.extensions import BatchGrad

# from functorch import grad_and_value, make_functional, vmap

class ModelTrainerCLS(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):

        print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(args.epsilon, args.delta))
        sampling_prob=args.batch_size/50000
        steps = int(args.epochs/sampling_prob)
        sigma, eps = get_sigma(sampling_prob, steps, args.epsilon, args.delta, rgp=False)
        noise_multiplier = sigma
        print('noise scale: ', noise_multiplier, 'privacy guarantee: ', eps)


        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                if args.enable_dp_ldp and (args.mechanism_type == "DP-SGD-laplace" or args.mechanism_type == "DP-SGD-gaussian"):
                    for param in model.parameters():
                        param.grad_sample = torch.zeros_like(param.data)

                    # Compute gradients for each sample in the batch
                    for sample_idx in range(x.size(0)):
                        model.zero_grad()
                        sample_x = x[sample_idx].unsqueeze(0)
                        sample_y = labels[sample_idx].unsqueeze(0)
                        log_probs = model(sample_x)
                        sample_loss = criterion(log_probs, sample_y)
                        sample_loss.backward()

                        # Accumulate gradients
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad_sample += param.grad / x.size(0)  # Averaging the gradients

                    # Clip and add noise
                    for param in model.parameters():
                        if param.grad_sample is not None:
                            torch.nn.utils.clip_grad_norm_(param, args.max_grad_norm)
                            noise = torch.normal(0, args.noise_multiplier * args.max_grad_norm, size=param.grad_sample.shape).to(device)
                            param.grad_sample += noise
                            param.grad = param.grad_sample
                            param.grad_sample = None  # Clear the intermediate gradient storage

                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = criterion(log_probs, labels)  # pylint: disable=E1102
                    loss.backward()

                    optimizer.step()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                step_loss = loss.item()

                batch_loss.append(step_loss)
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            ) 

    def train_iterations(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_steps = 0
        current_epoch = 0
        while current_steps < args.local_iterations:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
                current_steps += 1
                if current_steps == args.local_iterations:
                    break
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                target = target.long()
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                step_loss = loss.item()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += step_loss * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
