import torch
from torch import nn
import torch.optim as optim
from ...core.alg_frame.client_trainer import ClientTrainer
import logging
import logging

import numpy as np


# from functorch import grad_and_value, make_functional, vmap

class ModelTrainerCLS(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def cal_sensitivity(lr, clip, dataset_size):
        return 2 * lr * clip / dataset_size

    def calculate_noise_scale(self):
        if self.args.mechanism_type == "DP-SGD-laplace":
            epsilon_single_query = self.args.epsilon / self.args.epochs
            return 1 / epsilon_single_query
        elif self.args.mechanism_type == "DP-SGD-gaussian":
            epsilon_single_query = self.args.epsilon / self.args.epochs
            delta_single_query = self.args.delta / self.args.epochs
            return np.sqrt(2 * np.log(1.25 / delta_single_query)) / epsilon_single_query

    def per_sample_clip(self,net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def clip_gradients(self, net):
        if self.args.mechanism_type == "DP-SGD-laplace":
            # Laplace use 1 norm
            self.per_sample_clip(net, self.args.max_grad_norm, norm=1)
        elif self.args.mechanism_type == "DP-SGD-gaussian":
            # Gaussian use 2 norm
            self.per_sample_clip(net, self.args.max_grad_norm, norm=2)

    def add_noise(self, net, device):
        noise_scale = self.calculate_noise_scale()
        sensitivity = self.cal_sensitivity(self.args.learning_rate, self.args.max_grad_norm, self.args.batch_size)
        state_dict = net.state_dict()
        if self.args.mechanism_type == "DP-SGD-laplace":
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * noise_scale,
                                                                    size=v.shape)).to(device)
        elif self.args.mechanism_type == "DP-SGD-gaussian":
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * noise_scale,
                                                                   size=v.shape)).to(device)
        net.load_state_dict(state_dict)

    def train(self, train_data, device, args):

        # print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(args.epsilon, args.delta))
        # sampling_prob=args.batch_size/50000
        # steps = int(args.epochs/sampling_prob)
        # sigma, eps = get_sigma(sampling_prob, steps, args.epsilon, args.delta, rgp=False)
        # noise_multiplier = sigma
        # print('noise scale: ', noise_multiplier, 'privacy guarantee: ', eps)


        model = self.model

        if not args.enable_dp_ldp or args.mechanism_type != "DP-SGD-laplace" or args.mechanism_type != "DP-SGD-gaussian":
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
        mini_batch_size = 8
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data): 
                loss = 0
                losses = 0
                if args.enable_dp_ldp and (args.mechanism_type == "DP-SGD-laplace" or args.mechanism_type == "DP-SGD-gaussian"):
                    model.zero_grad()
                    total_grads = [torch.zeros(size=param.shape).to(self.args.device) for param in model.parameters()]
                
                # Initialize accumulated gradients for each parameter
                    # for param in model.parameters():
                    #     param.accumulated_grads = torch.zeros_like(param.data)

                    # Divide the main batch into mini-batches
                    for start in range(0, x.size(0), mini_batch_size):
                        end = start + mini_batch_size
                        x_mini, labels_mini = x[start:end].to(device), labels[start:end].to(device)

                        # Compute gradients for each mini-batch
                        # optimizer.zero_grad()
                        output = model(x_mini)
                        loss = criterion(output, labels_mini)
                        loss.backward()

                        self.clip_gradients(model)
                        grads = [param.grad.detach().clone() for param in model.parameters()]
                        for idx, grad in enumerate(grads):
                            total_grads[idx] += torch.mul(torch.div((mini_batch_size), len(x)), grad)
                        losses += loss.item() * mini_batch_size

                    for i, param in enumerate(model.parameters()):
                        param.grad = total_grads[i]

                        # Accumulate gradients
                    #     for param in model.parameters():
                    #         if param.grad is not None:
                    #             param.accumulated_grads += param.grad

                    # # Clip and add noise to the accumulated gradients
                    # for param in model.parameters():
                    #     if param.accumulated_grads is not None:
                    #         # Clip the gradients
                    #         clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                    #             param.accumulated_grads, args.max_grad_norm
                    #         )
                    #         # Add noise
                    #         noise = torch.normal(
                    #             mean=0,
                    #             std=args.noise_multiplier,
                    #             size=param.accumulated_grads.shape
                    #         ).to(device)
                    #         param.grad = param.accumulated_grads / x.size(0) + noise  # Averaging gradients

                    optimizer.step()
                    # optimizer.zero_grad()

                    self.add_noise(model, device)

                else:
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
                step_loss = 0
                if args.enable_dp_ldp and (args.mechanism_type == "DP-SGD-laplace" or args.mechanism_type == "DP-SGD-gaussian"):
                    step_loss = losses
                else:
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
