import fedml
from fedml import FedMLRunner
import sys
from argparse import ArgumentParser

def main(args):
    # init FedML framework
    args = fedml.init(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, None, dataset, model)
    fedml_runner.run()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/fedml_config.yaml")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--client_optimizer", type=str, default="sgd")
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gmf", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dampening", type=float, default=0.0)
    parser.add_argument("--nesterov", type=bool, default=False)
    args = parser.parse_args()

    main(args)

