import torch
from torch import nn
import torch.optim as optim
from ...core.alg_frame.client_trainer import ClientTrainer
import logging
import logging

import numpy as np


# from functorch import grad_and_value, make_functional, vmap

class ModelTrainerCLS(ClientTrainer):

    large_laplace_noise = None

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    
    def cal_sensitivity(self,lr, clip, dataset_size):
        return 2 * lr * clip / dataset_size
    
    def calculate_noise_scale(self):
        if self.args.mechanism_type == "DP-SGD-laplace":
            epsilon_single_query = self.args.epsilon
            return 1 / epsilon_single_query
        elif self.args.mechanism_type == "DP-SGD-gaussian":
            epsilon_single_query = self.args.epsilon
            delta_single_query = self.args.delta
            return np.sqrt(2 * np.log(1.25 / delta_single_query)) / epsilon_single_query
    
    def laplace_noise(self, shape, loc, scale):
        if ModelTrainerCLS.large_laplace_noise == None:
            logging.info("Creating large batch")
            gen = torch.distributions.Laplace(loc, scale)
            ModelTrainerCLS.large_laplace_noise = gen.sample(1000000)
        
        
        total_size = torch.Size(shape).numel()
        return ModelTrainerCLS.large_laplace_noise[:total_size].clone().reshape(shape) 

    def train(self, train_data, device, args):

        # print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(args.epsilon, args.delta))
        # sampling_prob=args.batch_size/50000
        # steps = int(args.epochs/sampling_prob)
        # sigma, eps = get_sigma(sampling_prob, steps, args.epsilon, args.delta, rgp=False)
        # noise_multiplier = sigma
        # print('noise scale: ', noise_multiplier, 'privacy guarantee: ', eps)


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
        mini_batch_size = 8
        sensitivity = self.cal_sensitivity(self.args.learning_rate, self.args.max_grad_norm, self.args.batch_size)
        noise_scale = self.calculate_noise_scale()
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data): 
                loss = 0
                if args.enable_dp_ldp and (args.mechanism_type == "DP-SGD-laplace" or args.mechanism_type == "DP-SGD-gaussian"):                
                # Initialize accumulated gradients for each parameter
                    for param in model.parameters():
                        param.accumulated_grads = torch.zeros_like(param.data, device=device)

                    # Divide the main batch into mini-batches
                    for start in range(0, x.size(0), mini_batch_size):
                        end = start + mini_batch_size
                        x_mini, labels_mini = x[start:end].to(device), labels[start:end].to(device)

                        # Compute gradients for each mini-batch
                        # optimizer.zero_grad()
                        output = model(x_mini)
                        loss = criterion(output, labels_mini)
                        loss.backward()

                        if args.mechanism_type == "DP-SGD-laplace":
                            for param in model.parameters():
                                if param.accumulated_grads is not None:
                                    # Clip the gradients
                                    clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                                        param.accumulated_grads, args.max_grad_norm, norm_type=1.0
                                    )
                        elif args.mechanism_type == "DP-SGD-gaussian":
                            for param in model.parameters():
                                if param.accumulated_grads is not None:
                                    # Clip the gradients
                                    clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                                        param.accumulated_grads, args.max_grad_norm
                                    )
                        

                        # Accumulate gradients
                        for param in model.parameters():
                            if param.grad is not None:
                                param.accumulated_grads += param.grad

                        # Clip and add noise to the accumulated gradients
                        
                        # logging.info("sensitivity * noise_scale= " + str(sensitivity*noise_scale))
                        for param in model.parameters():
                            if param.accumulated_grads is not None:
                                # # Clip the gradients
                                # clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                                #     param.accumulated_grads, args.max_grad_norm
                                # )
                                # # Add noise
                                noise = []
                                if args.mechanism_type == "DP-SGD-laplace":
                                    # Create a Laplace distribution with mean and std
                                    # laplace_dist = np.random.laplace(loc=0, scale=sensitivity*noise_scale, size=param.accumulated_grads.shape)
                                    # Sample noise from the distribution
                                    # noise = torch.from_numpy(laplace_dist).to(device)

                                    noise = self.laplace_noise(param.accumulated_grads.shape, loc=0, scale=sensitivity * noise_scale).to(device)
                                elif args.mechanism_type == "DP-SGD-gaussian":
                                    noise = torch.normal(
                                        mean=0,
                                        std=sensitivity*noise_scale,
                                        size=param.accumulated_grads.shape,
                                        device=device
                                    )
                                # if isinstance(noise, list):
                                #     noise = torch.tensor(noise, device=device)
                                # if noise.shape != param.accumulated_grads.shape:
                                #     noise = noise.view_as(param.accumulated_grads)
                                param.grad = param.accumulated_grads / x.size(0) + noise.to(dtype=param.accumulated_grads.dtype)  # Averaging gradients

                    optimizer.step()
                    model.zero_grad()

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
