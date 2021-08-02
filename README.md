## Environment Requirement
The code has been tested running under Python 3.5. The required packages are as follows:
* pytorch
* numpy
* scipy
* scikit-learn

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see main function in model/main.py).
* FM dataset
'cd model'
'python main.py' 
or
'cd model'
'python main.py --dataset music --dim 64 --l2_weight_rs 0.00005 --lr_rs 0.01 --batch_size 512 --n_epochs 50 --n_memory 16 --use_cuda True --n_neighbor 4 --kg_weight 0.0001 --dropout 0.3'
when you run codes in first way, you should tune hyperparameters in main.py

## Dataset (data/music)
* `train_data.txt(.npy)`
  * Train file.
  * Each line is a user with her/his positive sample and negative sample: (`userID`  `positive itemID` 'negative itemID').
  * Note that we take five times more items as negative samples for each positive interaction.  

* `test_data.txt(.npy)`
  * Test file.
  * Each line is a user with her/his positive sample and negative sample: (`userID`  `positive itemID` 'negative itemID').
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.

* `eval_data.txt(.npy)`
  * Test file.
  * Each line is a user with her/his positive sample and negative sample: (`userID`  `positive itemID` 'negative itemID').

* `kg_final.txt(.npy)`
  * knowledge graph file.
  * Each line is a triple: (`head entity ID`  `relation ID` 'tail entity ID').

* `kg_train.npy`
  * knowledge graph file(with negative sample).
  * Each line is a : (`head entity ID`  `relation ID` 'positive tail entity ID' 'negative tail entity ID').

* `adj_entity_gb.npy`
  * global neighbor entites file(with negative sample).
  * Each line are 50 global neighbor entites that are sampled in bias random walk:('entity ID'...)
