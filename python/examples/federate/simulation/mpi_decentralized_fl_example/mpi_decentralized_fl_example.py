import fedml
from fedml import FedMLRunner
import wandb

if __name__ == "__main__":

    wandb.login(key='63a4dccbf22454ee89c03213ddaba326fa6d7460')
    # init FedML framework
    args = fedml.init()

    # start training
    fedml_runner = FedMLRunner(args, None, None, None)
    fedml_runner.run()
