ETHUSDT futures trading using a Rules-Based Reinforcement Learning Model.

Agent module also includes a DQN agent and a Decision Tree agent, but the Rules-Based agent gives the most consistently profitable performance according to my tests.


Instructions:

To prep data for model training, go to the data folder `cd data` and run:
```
python3 prep_data.py
```

To train the model, go to the root of the project directory and run:
```
python3 -m scripts.train
```

To test the model, go to the root of the project directory and run:
```
python3 -m scripts.test
```
