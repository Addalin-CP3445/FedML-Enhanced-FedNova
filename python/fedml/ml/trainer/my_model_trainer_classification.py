import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
import logging
import copy
import logging

import numpy as np
from torch.optim.optimizer import Optimizer
from torch.distributions.laplace import Laplace
import math

# from functorch import grad_and_value, make_functional, vmap

class DP_SGD(Optimizer):
    def __init__(self, params, lr=0.01, clip_norm=1.0, noise_multiplier=1.0, batch_size=64, device='cpu', type='gaussian'):
        defaults = dict(lr=lr, clip_norm=clip_norm, noise_multiplier=noise_multiplier, batch_size=batch_size, device=device)
        super(DP_SGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            clip_norm = group['clip_norm']
            noise_multiplier = group['noise_multiplier']
            batch_size = group['batch_size']
            device = group['device']
            
            # Aggregate gradients
            grad_norms = []
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_norms.append(p.grad.data.norm(2))

            # Clip gradients
            total_norm = torch.stack(grad_norms).norm(2)
            clip_coef = clip_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.grad.data.mul_(clip_coef)

            # Add noise
            noise = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                if type == 'laplace':
                    scale = noise_multiplier
                    laplace_dist = Laplace(0, scale)
                    noise = laplace_dist.sample(p.grad.size()).to(device)
                elif type == 'gaussian':
                    noise = torch.normal(0, noise_multiplier, p.grad.size(), device=device)
                p.grad.data.add_(noise)

            # Apply gradients
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(d_p, alpha=-lr)

        return loss

class ModelTrainerCLS(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            if args.enable_dp_ldp and args.mechanism_type == "DP-SGD-gaussian":
                optimizer = DP_SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=args.learning_rate,
                    clip_norm=args.clip_norm,
                    noise_multiplier=args.clip_norm * math.sqrt(2 * math.log(1.25 / args.delta)) / args.epsilon,
                    batch_size=args.batch_size,
                    device=device,
                    type = 'gaussian'
                )
            elif args.enable_dp_ldp and args.mechanism_type == "DP-SGD-laplace":
                optimizer = DP_SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=args.learning_rate,
                    clip_norm=args.clip_norm,
                    noise_multiplier=args.clip_norm / args.epsilon,
                    batch_size=args.batch_size,
                    device=device,
                    type = 'laplace'
                )
            else:
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

                batch_loss.append(loss.item())
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

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                target = target.long()
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
