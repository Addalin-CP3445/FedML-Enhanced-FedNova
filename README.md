# FedML-Enhanced-FedNova

This FedML source code is derived from https://github.com/FedML-AI/FedML

Here Changes are made to experiment on FedAvg. FedProx and FedNova for my Masters research project 

Steps to reproduce the results:
- Clone the code
- Navigate to the ‘python’ directory and execute ‘pip install –e .’
- To run the various simulation for this project navigate first to the with_dp branch and then to the ‘python/examples/federate/simulation’ path
- Here inside each folder is the there is a config folder with fedml_config.yaml file that holds the configurations for the simulations, they can be edited to run for each simulation presented in the paper by using the following command inside the project simulation folder ‘sh run.sh 5 config/fedml_config.yaml’
- The results can be seen in the Wandb website that is configured for in the fedml_config.yaml file

