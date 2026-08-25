
# Information Relaying Environment 

Information relaying game: collaborative game in which agents control their motion and antennas to relay one message from a transmitting base station (blue square) to a receiving base (green square), preferably as fast as possible with minimal movement. 
Four scenarios are considered given by the combinations of isotropic or directed data links with the presence or absence of a jammer.

The repository contains the information relaying environment simulator, a multi-agent reinforcement learning (MARL) framework ([BenchMARL](https://github.com/facebookresearch/BenchMARL)) for training agents to solve the information relaying problem, and hand-crafted baseline. 

![demo gif](Media/Animations/Baseline/Trajectories/Comparison_all_scenarios/baseline_K5_row1_all_scenarios.gif)

## Setting Up the Simulator with BenchMARL

Training the Information Relay environment is currently easiest through [BenchMARL](https://github.com/facebookresearch/BenchMARL).

---

### 1. Clone the repository

Clone the Information Relaying repository together with the BenchMARL submodule:

```bash
git clone --recurse-submodule url/to/info_relay_env
```

Alternatively, clone without submodules and initialize them manually afterward:

```bash
cd Information-Relaying

git submodule init
git submodule update
```

---

### 2. Create a virtual environment

Use **Python 3.11**:

```bash
python3.11 -m venv path/to/venv
```

Activate the virtual environment:

```bash
source path/to/venv/bin/activate
```

---

### 3. Install BenchMARL and dependencies

The `BenchMARL` folder is located inside `Info-relay-implementation`.

Install BenchMARL in editable mode:

```bash
pip install -e BenchMARL/
```

Install the remaining requirements:

```bash
pip install -r Info-relay-implementation/benchmarl_integration/requirements.txt
```

> **Note:**  
> Some package versions installed here will overwrite versions provided by BenchMARL. This is expected.

---

### 4. Start training

When training you can choose which algorithm to use. In the paper we used MAPPO and MADDPG. Run training with:

```bash
python BenchMARL/benchmarl/run.py algorithm=mappo task=customenv/info_relay
```

It is also possible to do so-called multiruns with several different tasks or algorithms. One could for example queue several different parameter choices by using different info_relay.yaml files:

```bash
python BenchMARL/benchmarl/run.py -m algorithm=mappo/maddpg task=customenv/info_relay,info_relay2
```

---

## Configuration

The hyperparameters and environment parameters are changed inside configuration files which are located in the `Info-relay-implementation/BenchMARL/benchmarl/conf` folder.

### Training parameters

Edit `experiment/base_experiment.yaml` to modify:
- Learning rate (`lr`)
- Number of training episodes
- Batch sizes
- Other experiment hyperparameters

### Environment parameters

Edit `task/customenv/info_relay.yaml` to modify:
- Number of agents
- Discrete or continous actions
- Jammer settings
- Communication settings
- Other environment-specific parameters

---

## Environment files and symbolic links

The code for the Information relay environment is mainly located inside `info_relay_env_v2.py` and `info_relay_classes.py` files. The Information relay environment files inside the `BenchMARL` directory are symbolic links to the files in this repository.

Any changes made in this repository are therefore automatically reflected inside BenchMARL.

---

## Running the environment with pre-trained agent
Make sure the repo is cloned with git lfs installed "git lfs install".
    You will then receive the outputs/ folder containing checkpoints.
    Run 
    ```bash
    python BenchMARL/benchmarl/evaluate.py path/to/checkpoint_file.pt
    ```

    To run on the evaluation set (using the same inital 10000 conditions each time it is run), change the default
    parameter boolean pre_determined_scenario in info_relay_env_v2.py to True.

---

## Evaluating the results
The scripts used to generate the plots and calculate the evalution metrics are located inside `Evaluation/evaluate.py`. To run the different scripts, choose eval_mode inside `main()`. `eval_mode = 16` results generates the heatmaps from the paper. 

---

## Running the baseline
The baseline takes a scene (an initial state defining agent positions and antenna orientations, and distance between bases, and one of the four scenarios) and returns a solution to the relaying problem as a full state trajectory of all agents. 
This can be done in the following steps:

---

1. **Set up environment parameters:**
   ```python
   from Baseline.baseline import baseline, sample_scenario, load_premade_scenario
   
   Rcom = 1.0        # Communication range
   sigma = 0.2       # Maximum agent displacement per time step
   beta = 0.99       # Discount factor
   c_pos = 0.5       # Cost coefficient for agent movement
   c_phi = 0.1       # Cost coefficient for antenna steering
   ```

2. **Load or generate a scene:**
   ```python
   # Option A: Load a premade scene 
   sample = load_premade_scenario(option=1, Rcom=Rcom, sigma=sigma)
   
   # Option B: Sample a random scenario
   # sample = sample_scenario(K=5, Rcom=Rcom, R=5.0, Ra=0.6, sigma=sigma, seed=42)
   ```

3. **Configure scenario parameters:**
   ```python
   directed = False          # True for directional antenna transmission
   phi_agents = sample['phi_agents'] if directed else None
   
   jammer_info = None        # Set jammer parameters for jammed scenarios
   # For jammed scenario, use: jammer_info = {'p_jammer': sample['p_j'], 'v_jammer': sample['dp_j']}
   ```

4. **Run the baseline solver:**
   ```python
   result = baseline(
       p_agents=sample['p_agents'],
       p_tx=sample['p_tx'],
       p_recv=sample['p_recv'],
       Rcom=Rcom,
       sigma=sigma,
       beta=beta,
       c_pos=c_pos,
       c_phi=c_phi,
       phi_agents=phi_agents,
       jammer_info=jammer_info
   )
   ```

5. **Extract the solution:**
   ```python
   # Key outputs from result dictionary:
   path = result['path']                    # Relay path (agent sequence)
   relay_points = result['relay_points']    # Positions where agents relay
   p_trajectories = result['p_trajectories'] # Agent position trajectories over time
   value = result['value']                  # Solution quality metric (budget - cost)
   delivery_time = result['delivery_time']  # Time steps to deliver message
   ```

6. **Visualize the solution:**
    ```python
    from Evaluation.evaluate import plot_trajectory, animate_trajectory
    
    # Option A: Static plot of agent trajectories
    plot_trajectory(
       result['p_trajectories'],
       sample['p_tx'],
       sample['p_recv'],
       sample['Rcom'],
       savepath=None  # Set to a path string to save as image
    )
    
    # Option B: Animation of agent trajectories over time
    animate_trajectory(
       result['p_trajectories'],
       sample['p_tx'],
       sample['p_recv'],
       sample['Rcom'],
       savepath='baseline_animation.gif'  # Set to None to display without saving
    )
    ```

## Citation
Please consider citing if you found this repository helpful.
```
@article{persson2025dynamic,
  title={Dynamic one-time delivery of critical data by small and sparse UAV swarms: a model problem for MARL scaling studies},
  author={Persson, Mika and Lidman, Jonas and Ljungberg, Jacob and Sandelius, Samuel and Andersson, Adam},
  journal={arXiv preprint arXiv:2512.09682},
  year={2025}
}
```

