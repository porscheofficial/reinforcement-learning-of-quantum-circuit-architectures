# Reinforcement learning of quantum circuit architectures for molecular potential energy curves
[![arXiv](https://img.shields.io/badge/arXiv-2511.16559-b31b1b.svg)](https://arxiv.org/abs/2511.16559)

### Abstract
Quantum chemistry and optimization are two of the most prominent applications of quantum computers. Variational quantum algorithms have been proposed for solving problems in these domains. However, the design of the quantum circuit ansatz remains a challenge. Of particular interest is developing a method to generate circuits for any given instance of a problem, not merely a circuit tailored to a specific instance of the problem. To this end, we present a reinforcement learning (RL) approach to learning a problem-dependent quantum circuit mapping, which outputs a circuit for the ground state of a Hamiltonian from a given family of Hamiltonians. For quantum chemistry, our RL framework takes as input a molecule and a discrete set of bond distances, and it outputs a bond-distance-dependent quantum circuit that is generalizable to arbitrary bond distances along the potential energy curve. The inherently non-greedy approach of our RL method contrasts with existing greedy approaches to adaptive, problem-tailored circuit constructions. We demonstrate its effectiveness for the four-qubit and six-qubit lithium hydride molecules, as well as an eight-qubit H4 chain. Our learned mapping and circuits are interpretable in a physically meaningful manner, paving the way for applying our framework to the development of novel quantum circuits for the ground states of large-scale molecular systems. 

## Code Organisation
This repository contains the source code for the Reinforcement Learning Framework used to construct quantum circuit ansätze as described in [Paper Title] (add link if available). The modules of the framework,to execute the settings discussed in the paper, are organized as follows:

* **training.py**: Central controller for the reinforcement learning process, calling all subprocesses in the framework.

* **SAC.py**: Implements the Soft Actor-Critic (SAC) algorithm and thus manages the update process of the neural networks.

* **QuantumStateEnv.py**: Reinforcement learning environment for building quantum circuits by applying gates and evaluating resulting states and energies.

* **reward.py**: Defines the reward function used in the reinforcement learning process.

* **ReplayBuffer.py**:  Manages the storage of transitions and samples batches for updates.

* **NeuralNet.py**: Defines the neural network architecture for the actor and critic networks.
  
* **hamiltonians/Ham_gen.py:** Generates the Hamiltonians for LiH and H4 using Qiskit Nature and Tequila.

* **hamiltonians/JSP.py:** Generates the Hamiltonians for the job shop scheduling problem.
  
*  **utils.py**: Provides a collection of auxiliary functions for various tasks.
  
*  **Plots.py:** Automatically plots the training outcome for an initial overview.

*  **predict.py:** Used after training to generate the predicted potential energy curve.
  
  
*  **config/:** Contains configuration files for each setup discussed in the paper.
    - **config_1.py:** Configuration file for training four-qubit LiH at the bond distance 2.2 Å.
    - **config_2.py:** Configuration file for training four-qubit LiH on the bond distance range from 1.0 Å to 4.0 Å with a step size of 0.1 Å.
    - **config_3.py:** Configuration file for training six-qubit LiH at the bond distance 2.2 Å.
    - **config_4.py:** Configuration file for training six-qubit LiH on the bond distance range from 1.0 Å to 4.0 Å with a step size of 0.1 Å.
    - **config_5.py:** Configuration file for training H4 at the bond distance 1.5 Å.
    - **config_6.py:** Configuration file for training H4 on the bond distance range from 0.5 Å to 1.6 Å with a step size of 0.1 Å.
    - **config_7.py:** Configuration file for training the JSP.
    
## Getting started
This code is implemented in Python 3.11.2 and optimized for execution with CUDA support (validated with CUDA version 12.6). A CPU-only execution is also possible, but not recommended for efficient performance.

### Step 1: Clone the Repository
```
git clone https://github.com/porscheofficial/reinforcement-learning-of-quantum-circuit-architectures
```

### Step 2: Set Up the Environment
```
python3 -m venv RLVQEenv
source RLVQEenv/bin/activate
```

### Step 3: Install Required Packages
```
pip install -r requirements.txt
```


### Step 4: Run the Code
```
python3 training.py config_1.py
```

### Step 5: The output of training.py

After the training session x is completed, the resulting data is saved in the following directory structure:

`results/<your_choice_of_folder_name>/<timestamp_folder>/training_session_x/`

For each training session, the following files are saved:

- **`results_x.py`**  
  
  ```
  results=np.load('results_x.npy', allow_pickle=True)
  episode = list(zip(*results))[0]
  energy = np.real(list(zip(*results))[1])
  evaluation = np.real(list(zip(*results))[2])
  reward = list(zip(*results))[3]
  circuit= np.real(list(zip(*results))[4])
  parameters = np.real(list(zip(*results))[5])
  bond_distance=np.real(list(zip(*results))[6])
  ```
   
- **`rl_quantities_x.py`**
  ```
  rl_quantities=np.load('rl_quantities_x.npy', allow_pickle=True)
  episode = list(zip(*rl_quantities))[0]
  policy_loss = list(zip(*rl_quantities))[1]
  Q_loss = list(zip(*rl_quantities))[2]
  alpha_d = list(zip(*rl_quantities))[3]
  alpha_c = list(zip(*rl_quantities))[4]
  entropy_d = list(zip(*rl_quantities))[5]
  entropy_c = list(zip(*rl_quantities))[6]
  ```

- **`actor_model_x.pt`**  
  The policy that can be reused for generalization for unseen bond distances.


### Step 6: The prediction 
You can run predictions by executing the following command, specifying the same configuration file used during training and
the path to the corresponding result folder:
```
python predict.py config_1.py  /full/path/to/results_folder
```
The prediction results will be automatically saved as:
`training_session_{session}/unseen_prediction_{session}.npy`

### Step 7: Plotting results
Jupyter notebooks and corresponding molecular data for plotting single-distance results and potential energy curves (PECs) can be found in the visualization folder.

## ✨ Contributing

The Porsche Open Source Platform is openly developed in the wild and contributions (both internal and external) are highly appreciated. See [CONTRIBUTING.md](CONTRIBUTING.md) on how to get started.

If you have feedback or want to propose a new feature, please [open an issue](https://github.com/porscheofficial/reinforcement-learning-of-quantum-circuit-architectures/issues/new), which will then be monitored/prioritized in our open [GitHub Project board].

## 🙌 Acknowledgements

This project is a joint initiative of [Porsche AG](https://www.porsche.com/) and [Porsche Digital](https://www.porsche.digital/).

## ✒️ License

Copyright © 2025 Dr. Ing. h.c. F. Porsche AG

Dr. Ing. h.c. F. Porsche AG publishes the Porsche Open Source Platform software and accompanied documentation (if any) subject to the terms of the [MIT license](https://opensource.org/licenses/MIT). All rights not explicitly granted to you under the MIT license remain the sole and exclusive property of Dr. Ing. h.c. F. Porsche AG.

Apart from the software and documentation described above, the texts, images, graphics, animations, video and audio files as well as all other contents on this website are subject to the legal provisions of copyright law and, where applicable, other intellectual property rights. The aforementioned proprietary content of this website may not be duplicated, distributed, reproduced, made publicly accessible or otherwise used without the prior consent of the right holder.
