
# Reinforcement Learning for Solving the Traveling Salesman Problem (TSP)

We use Reinforcement for solving Travelling Salesman Problem (TSP)

## Paper
Base code inspired by the implementation of paper: [Reinforcement Learning for Solving the Vehicle Routing Problem](https://arxiv.org/abs/1802.04240v2). 

## Information
### Currently under constructions. Modifications needed:
1. The model isn't being saved, therefore the variables arent being reused in the correct way (I think)
2. SOLVED Actor/Critic are not talking to each other and that's why variables never change
3. SOLVED Standard deviation is null but that's because there's no data to compare it to.
4. SOLVED Never makes a decision, always returns the same value.
5. We have to save the model and print the results
6. Evaluate the model to see performance
7. See how to run on cloud - LIACS servers
8. See tests for the model on Readme, test all parameters

## Dependencies

* Pytorch 2.2.2
* Numpy

## How to Run
### Train
By default, the code is running in the training mode on a single gpu. For running the code, one can use the following command:
```bash
python main.py --task=vrp10
```

It is possible to add other config parameters like:
```bash
python main.py --task=vrp10 --gpu=0 --n_glimpses=1 --use_tanh=False 
```
There is a full list of all configs in the ``config.py`` file. Also, task specific parameters are available in ``task_specific_params.py``
### Inference
For running the trained model for inference, it is possible to turn off the training mode. For this, you need to specify the directory of the trained model (to-do), otherwise random model will be used for decoding:
```bash
python main.py --task=vrp10 --is_train=False --model_dir=./path_to_your_saved_checkpoint
```
The default inference is run in batch mode, meaning that all testing instances are fed simultanously. It is also possible to do inference in single mode, which means that we decode instances one-by-one. The latter case is used for reporting the runtimes and it will display detailed reports. For running the inference with single mode, you can try:
```bash
python main.py --task=vrp10 --is_train=False --infer_type=single --model_dir=./path_to_your_saved_checkpoint
```
### Logs
All logs are stored in ``result.txt`` file stored in ``./logs/task_date_time`` directory.

## Acknowledgements
Thanks to [pemami4911/neural-combinatorial-rl-pytorch](https://github.com/pemami4911/neural-combinatorial-rl-pytorch) for getting the idea of restructuring the code.
