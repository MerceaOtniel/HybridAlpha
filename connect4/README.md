# Connect4 implementation for Alpha Zero General

Alpha-zero general implementation of connect4.
Neural network architecture was copy-pasted from the game of Othello, so could likely be improved.

To train a model for Connect4, update the imports in ```main.py``` to:
```python
from Coach import Coach
from connect4.Connect4Game import Connect4Game
from connect4.tensorflow.NNet import NNetWrapper as nn
from utils import dotdict
```

and the first line of ```__main__``` to
```python
g = Connect4Game()
```

Make similar changes to ```pit.py```.

To start training a model for Connect4:
```bash
python main.py
```
To start a tournament of 100 episodes with the model-based player against a random player:
```bash
python pit.py
```
You can play againt the model by switching to HumanPlayer in ```pit.py```

