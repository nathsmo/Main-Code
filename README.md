
# Reinforcement Learning for Solving the Traveling Salesman Problem (TSP)

We use Reinforcement for solving Travelling Salesman Problem (TSP)

## Paper
Base code inspired by the implementation of paper: [Reinforcement Learning for Solving the Vehicle Routing Problem](https://arxiv.org/abs/1802.04240v2). 

## Information
## Currently under construction!!!! 
*Ideas to implement to make code more efficient*
* Different Embedding
* Self Attention (via Multi-head Attention)


**Ultimate question for our angle:**  
* What is the factor taking too long to train the model. 
* Reduce Runtime

## Dependencies

* Pytorch 2.2 or > 
* Numpy
* Python 3.11.7 or >

## How to Run
### Train
By default, the code is running in the training mode on a single gpu. For running the code, one can use the following command:

```bash
# DPN_10: Basic Pointer Network - Decoder
python main.py --variation='DPN_10_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv 

# DSA_1H_10: Self-Attention Decoder (1 head)
python main.py --variation='DSA_1H_10_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1
```

There is a full list of all configs in the ``config.py`` file. Also, task specific parameters are available in ``task_specific_params.py``

### Inference
For running the trained model for inference, it is possible to turn off the training mode. For this, you need to specify the directory of the trained model (to-do), otherwise random model will be used for decoding:
```bash
python main.py --task=tsp10 --is_train=False --model_dir=./path_to_your_saved_checkpoint
```
The default inference is run in batch mode, meaning that all testing instances are fed simultanously. It is also possible to do inference in single mode, which means that we decode instances one-by-one. The latter case is used for reporting the runtimes and it will display detailed reports. For running the inference with single mode, you can try:
```bash
python main.py --task=tsp10 --is_train=False --infer_type=single --model_dir=./path_to_your_saved_checkpoint
```
## Example for inference
```bash
python main.py --variation='DSA_1H_10_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H_10_conv/model/agent_complete.pth 
```
### Logs
All logs are stored in ``result.txt`` file stored in ``./logs/task_date_time`` directory.

<!---## Acknowledgements
Thanks to [pemami4911/neural-combinatorial-rl-pytorch](https://github.com/pemami4911/neural-combinatorial-rl-pytorch) for getting the idea of restructuring the code.)---!>

<!---
* Modify Reinforcement Learning - see options. (More efficient, see different reward functions, aim to lower time)
* Modify Neural Network Parameters
* Improve solution quality from Amazon dataset (compare to previous results from other papers) ---!>

