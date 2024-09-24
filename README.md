# Enhancing Accuracy in Decentralized Federated Learning while Preserving Differential Privacy

In this research, I explored the challenge of balancing model accuracy and privacy in decentralized
federated learning systems, a cutting-edge machine learning approach that enables collaborative model
training across multiple devices without sharing raw data. The project focused on the efficacy of FedNova, a
novel federated learning optimization method, in improving model accuracy while maintaining differential
privacy.

The study involved a comprehensive comparison of FedNova with two established baseline methods, FedAvg
and FedProx, under varying privacy constraints. Differential privacy was incorporated using two mechanisms:
Gaussian and Laplace noise, applied both to the input data and to model gradients during training. The
experiments were conducted on the CIFAR-10 dataset using the VGG11 convolutional neural network model
within the FedML framework.

Key findings revealed that FedNova, particularly when paired with differentially-private stochastic gradient
descent (DP-SGD), delivers markedly better results compared to when noise is added directly to the input
data, particularly in high-privacy scenarios. The research demonstrated that FedNova's normalized
aggregation method is particularly effective in managing data heterogeneity among clients, leading to
superior accuracy even under stringent privacy constraints.

This project contributes to the ongoing development of privacy-preserving federated learning systems,
offering insights into how advanced optimization techniques can be integrated with differential privacy to
achieve a better balance between privacy and model performance. The results have potential applications
in fields where data privacy is critical, such as healthcare and finance, and highlight the importance of
innovative approaches in the evolving landscape of machine learning and data science.

# Citations

- The base code is derived from https://github.com/FedML-AI/FedML
- The implementation of DP-SGD is derived from https://github.com/wenzhu23333/Differential-Privacy-Based-Federated-Learning
