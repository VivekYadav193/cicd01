# ml-ci-cd
Basic example of CI/CD in a ML project.

```
pip install --upgrade pip
pip install -r requirements.txt
```

```
python3 code/get_data.py
```

```
dvc init
```

```
dvc add data/train_data.csv
dvc add data/test_data.csv
```

```
dvc stage add -n featurization \
-d data/train_data.csv \
-d data/test_data.csv \
-d code/featurization.py \
-o data/norm_params.json \
-o data/processed_train_data.npy \
-o data/processed_test_data.npy \
python3 code/featurization.py
```

```
dvc stage add -n training \
-d code/train_model.py \
-d data/processed_train_data.npy \
-o data/model.pkl \
-m metrics/train_metric.json \
python3 code/train_model.py
```

```
dvc stage add -n eval \
-d code/eval.py \
-d data/model.pkl \
-d data/processed_test_data.npy \
-m metrics/eval.json \
python3 code/eval.py
```

```
dvc repro
```