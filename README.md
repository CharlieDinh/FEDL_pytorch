# Federated Learning over Wireless Networks: Convergence Analysis and Resource Allocation

This repository is for the Experiment Section of the paper:
"Federated Learning over Wireless Networks: Convergence Analysis and Resource Allocation"

Authors:
Canh T. Dinh, Nguyen H. Tran, Minh N. H. Nguyen, Choong Seon Hong, Wei Bao, Albert Zomaya, Vincent Gramoli

Link:
https://arxiv.org/abs/1910.13067

# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**

- The code can be run on any pc.

# Dataset: We use 3 datasets: MNIST, FENIST, and Synthetic

- To generate non-idd MNIST Data: In folder data/mnist,  run: "python3 generate_niid_mnist_100users.py" 
- To generate FEMNIST Data: first In folder data/nist run preprocess.sh to obtain all raw data, or can be download in the link below, then run  python3 generate_niid_femnist_100users.py
- To generate niid Linear Synthetic: In folder data/linear_synthetic, run: "python3 generate_linear_regession.py" 
- The datasets are available to download at: https://drive.google.com/drive/folders/1Q91NCGcpHQjB3bXJTvtx5qZ-TrIZ9WzT?usp=sharing


# Produce figures in the paper:
- There is a main file "main.py" which allows running all experiments and 3 files "main_mnist.py, main_nist.py, main_linear.py" to produce the figures corresponding for 3 datasets. It is noted that each experiment is run at least 10 times and then the result is averaged.
