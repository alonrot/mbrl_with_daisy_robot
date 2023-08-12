The default container used in `SASDataset` is set to `TensorList`. This can lead
to significant speedup compared to using Python's built-in `list` container:

```bash
$ python mbrl/tests/test_dataset.py # 300K elements in list
Profiling saving/loading dataset of 300000 tensors for 5 times ...
Took 19.28784 sec to save, 13.80702 sec to load [with container <class 'list'>]
Took  0.03220 sec to save,  0.24479 sec to load [with container <class 'mbrl.dataset.TensorList'>]
```
