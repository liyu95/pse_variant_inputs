# flash calulation with variant components using deep learning

## Current result:
#### CNN: not converge
#### FCNet
14->8: MSE: 0.017, PCC: 0.29 <br/>
8->14: MSe: 0.007, PCC: 0.4
#### AttentionNet
14->8: MSE: 0.018, PCC: 0.20 <br/>
8->14: MSe: 0.009, PCC: 0.5


## How to use the code
Create the environment:
```
conda env create -f environment.yml
```
Activate the environment:
```
Conda activate tf_14_py2
```
Download the data and put them into the data folder, then
```
cd code
python main.py
```
The *parameter.py* file is used to configure the hyperparameter of the model