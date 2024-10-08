import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor
from ....core.schedule.runtime_estimate import t_sample_fit
from ....core.schedule.seq_train_scheduler import SeqTrainScheduler


class FedNovaAggregator(object):
    def __init__(
        self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        server_aggregator,
    ):
        self.aggregator = server_aggregator

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.result_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.runtime_history = {}
        for i in range(self.worker_num):
            self.runtime_history[i] = {}
            for j in range(self.args.client_num_in_total):
                self.runtime_history[i][j] = []

        self.global_momentum_buffer = dict()

    def get_global_model_params(self):
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, local_result):
        logging.info("add_model. index = %d" % index)
        self.result_dict[index] = local_result
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def record_client_runtime(self, worker_id, client_runtimes):
        for client_id, runtime in client_runtimes.items():
            self.runtime_history[worker_id][client_id].append(runtime)

    def generate_client_schedule(self, round_idx, client_indexes):
        if hasattr(self.args, "simulation_schedule") and round_idx > 5:
            simulation_schedule = self.args.simulation_schedule
            fit_params, fit_funcs, fit_errors = t_sample_fit(
                self.worker_num,
                self.args.client_num_in_total,
                self.runtime_history,
                self.train_data_local_num_dict,
                uniform_client=True,
                uniform_gpu=False,
            )
            logging.info(f"fit_params: {fit_params}")
            logging.info(f"fit_errors: {fit_errors}")
            avg_fit_error = sum(client_error for gpu_errors in fit_errors.values() for client_error in gpu_errors.values()) / sum(len(gpu_errors) for gpu_errors in fit_errors.values())
            if self.args.enable_wandb:
                wandb.log({"RunTimeEstimateError": avg_fit_error, "round": round_idx})

            mode = 0
            workloads = np.array([self.train_data_local_num_dict[client_id] for client_id in client_indexes])
            constraints = np.array([1] * self.worker_num)
            memory = np.array([100])
            my_scheduler = SeqTrainScheduler(
                workloads, constraints, memory, fit_funcs, uniform_client=True, uniform_gpu=False
            )
            y_schedule, output_schedules = my_scheduler.DP_schedule(mode)
            client_schedule = [client_indexes[indexes] for indexes in y_schedule]
            logging.info(f"Schedules: {client_schedule}")
        else:
            client_schedule = np.array_split(client_indexes, self.worker_num)
        return client_schedule

    def get_average_weight(self, client_indexes):
        average_weight_dict = {}
        training_num = sum(self.train_data_local_num_dict[client_index] for client_index in client_indexes)

        for client_index in client_indexes:
            average_weight_dict[client_index] = self.train_data_local_num_dict[client_index] / training_num
        return average_weight_dict

    def fednova_aggregate(self, params, norm_grads, tau_effs, tau_eff=0):
        if tau_eff == 0:
            tau_eff = sum(tau_effs)
        cum_grad = norm_grads[0]
        for k in norm_grads[0].keys():
            for i in range(len(norm_grads)):
                if i == 0:
                    cum_grad[k] = norm_grads[i][k] * tau_eff
                    #cum_grad[k] = norm_grads[i][k].float() * tau_eff
                else:
                    cum_grad[k] += norm_grads[i][k] * tau_eff
                    #cum_grad[k] += norm_grads[i][k].float() * tau_eff
        for k in params.keys():
            #params[k] = params[k].float()
            if self.args.gmf != 0:
                if k not in self.global_momentum_buffer:
                    buf = self.global_momentum_buffer[k] = torch.clone(cum_grad[k]).detach()
                    buf.div_(self.args.learning_rate)
                else:
                    buf = self.global_momentum_buffer[k]
                    buf.mul_(self.args.gmf).add_(1 / self.args.learning_rate, cum_grad[k])
                logging.info("params[k]: {}".format(params[k]))
                logging.info("self.args.learning_rate: {}".format(self.args.learning_rate))
                params[k].sub_(self.args.learning_rate, buf.to(params[k].device))
            else:
                logging.info("params[k]: {}".format(params[k]))
                logging.info("cum_grad[k]: {}".format(cum_grad[k]))
                params[k].sub_(cum_grad[k].to(params[k].device))
        return params

    def aggregate(self):
        start_time = time.time()
        grad_results = []
        t_eff_results = []

        for idx in range(self.worker_num):
            if len(self.result_dict[idx]) > 0:
                for client_result in self.result_dict[idx]:
                    grad_results.append(client_result["grad"])
                    t_eff_results.append(client_result["t_eff"])
        logging.info("len of self.result_dict[idx] = " + str(len(self.result_dict)))

        init_params = self.get_global_model_params()
        logging.info("|||||||||||||||||||||||||||||||||||||||||||||||||init_params: {}".format(init_params))
        w_global = self.fednova_aggregate(init_params, grad_results, t_eff_results)
        self.set_global_model_params(w_global)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return w_global

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        logging.info("@@@@@@@@@@@@@@@@@@@@@@ round_idx: {}".format(round_idx))
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            metrics = [0,0]

            self.args.round_idx = round_idx
            if round_idx == self.args.comm_round - 1:
                metrics = self.aggregator.test(self.test_global, self.device, self.args)
            else:
                metrics = self.aggregator.test(self.val_global, self.device, self.args)
            
            # Log metrics to WandB

            logging.info("@@@@@@@@@@@@@@@@@@@metrics for wandb: {}".format(metrics))
            if self.args.enable_wandb:
                wandb.log({"Test/Acc Fednovaaggregator": metrics[0], "round": round_idx})
                wandb.log({"Test/Loss Fednovaaggregator": metrics[1], "round": round_idx})
