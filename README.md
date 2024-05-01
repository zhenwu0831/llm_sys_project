# LLM Systems Project: Implement DPO in MiniTorch

## Preparation

### Install requirements
```
pip install -r requirements.extra.txt
pip install -r requirements.txt
```

### Install minitorch
```
pip install -e .
```


```
bash compile_cuda.sh
```

## File Structure

This project is organized into several directories and files. Below is an overview of the key components:

- `data/`
  - `imdb.json` - This file contains the IMDb dataset with generated preference pairs.
  - `sampled_data.json` - A smaller subset (10%) of the `imdb.json` used for quicker testing and development.

- `project/`
  - `dpo.py` - Implementation of the Direct Preference Optimization (DPO) algorithm.
  - `generate_preference_pair.py` - Script to generate preference pairs from the IMDb dataset.
  - `no_dpo.py` - Baseline model implementation without DPO for comparison.

- `minitorch/`
  - `nn.py` - Modified neural network module that includes the DPO `preference_loss` function to support DPO in MiniTorch.

## Usage
### Generate the Preference Pair Dataset
To generate new preference pairs from the IMDb dataset, use:
```
python generate_preference_pair.py
```

### Running the DPO Model
To run the DPO model and conduct experiments, navigate to the `project/` directory and execute:
```
python dpo.py
```
To compare the results with the baseline model that does not use DPO, run:
```
python no_dpo.py
```
