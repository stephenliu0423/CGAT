# CGAT

This is the implementation for the following paper:

>Yong Liu, Susen Yang, Yonghui Xu, Chunyan Miao, Min Wu, and Juyong Zhang. "Contextualized Graph Attention Network for Recommendation with Item Knowledge Graph." IEEE Transactions on Knowledge and Data Engineering (2021).

## Environment Requirement
The code has been tested under Python 3.5. The required packages are as follows:
* pytorch
* numpy
* scipy
* scikit-learn

## Examples to Run the Codes
The instruction of commands can be found in the codes (see main function in model/main.py).
* python main.py    
* python main.py --dataset music --dim 64 --l2_weight_rs 0.00005 --lr_rs 0.01 --batch_size 512 --n_epochs 50 --n_memory 16 --use_cuda True --n_neighbor 4 --kg_weight 0.0001 --dropout 0.3 

## Dataset (data/music)
* `train_data.txt(.npy)`
  * Training file
  * Each line is a user with her/his positive sample and negative sample: (`userID`  `positive itemID` `negative itemID`).

* `test_data.txt(.npy)`
  * Testing file.

* `eval_data.txt(.npy)`
  * Validation file.

* `kg_final.txt(.npy)`
  * Knowledge graph file.
  * Each line is: (`head entity ID`  `relation ID` `tail entity ID`).

* `kg_train.npy`
  * Knowledge graph file(with negative sample).
  * Each line is: (`head entity ID`  `relation ID` `positive tail entity ID` `negative tail entity ID`).

* `adj_entity_gb.npy`
  * Global neighbor entites file(with negative sample).
  * Each line includes global neighbor entites sampled in bias random walk: (`entity ID`...)

## Citation

```
@article{liu2021contextualized,
  title={Contextualized Graph Attention Network for Recommendation with Item Knowledge Graph},
  author={Liu, Yong and Yang, Susen and Xu, Yonghui and Miao, Chunyan and Wu, Min and Zhang, Juyong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021},
  publisher={IEEE}
}
```
