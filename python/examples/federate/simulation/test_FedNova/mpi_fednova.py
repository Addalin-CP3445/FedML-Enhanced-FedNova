import fedml
from fedml import FedMLRunner
# import wandb

if __name__ == "__main__":

    # wandb.login(key='63a4dccbf22454ee89c03213ddaba326fa6d7460')
    # init FedML framework
    args = fedml.init()

    # init device
    # device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, None, dataset, model)
    fedml_runner.run()
