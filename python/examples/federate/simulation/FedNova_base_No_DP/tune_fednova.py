import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os

def train_federated(config):
    # Create command line arguments string
    cmd_args = f"--config=config/fedml_config.yaml --learning_rate={config['learning_rate']} --batch_size={config['batch_size']} --client_optimizer={config['client_optimizer']} --weight_decay={config['weight_decay']} --gmf={config['gmf']} --mu={config['mu']} --momentum={config['momentum']} --dampening={config['dampening']} --nesterov={config['nesterov']}"
    os.system(f"mpirun -np 3 --oversubscribe python mpi_fednova_ray_tune.py {cmd_args}")

search_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "client_optimizer": tune.choice(["sgd", "adam"]),
    "weight_decay": tune.loguniform(1e-5, 1e-3),
    "gmf": tune.choice([0.0, 0.5, 1.0]),
    "mu": tune.choice([0.0, 0.1, 0.5]),
    "momentum": tune.uniform(0.0, 0.9),
    "dampening": tune.uniform(0.0, 0.5),
    "nesterov": tune.choice([False, True])
}

scheduler = ASHAScheduler(
    metric="accuracy",
    mode="max",
    max_t=100,
    grace_period=10,
    reduction_factor=2
)

analysis = tune.run(
    train_federated,
    resources_per_trial={"cpu": 1, "gpu": 0},
    config=search_space,
    num_samples=10,
    scheduler=scheduler
)

best_config = analysis.best_config
print("Best hyperparameters found were: ", best_config)

with open("best_hyperparameters.txt", "w") as f:
    f.write(f"Best hyperparameters found were: {best_config}\n")
