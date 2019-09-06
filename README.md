# AlphaZero
HybridAlpha - a mix between AlphaGo Zero and AlphaZero for multiple games

This project has the goal of creating an Hybrid between AlphaZero and AlphaGo Zero, both published by DeepMind.

Moreover, this is an improved and extended implementation of the project which can be found here https://github.com/suragnair/alpha-zero-general. However, this project wants to copy as much as possible the algorithm provided by AlphaZero, being different in only certain key aspects which makes HybridAlpha more performant than AlphaZero when run on resource constrained systems.

# Ways in which this project improves over the repository presented above are:

- better heuristics for testing the networks

- the addition of Alpha-Beta pruning algorithms which for some of the games also takes into account the depth in the search tree to provide a stronger heuristics and to test the network capacity of generalization.

- the games start from random position when pitting the network, in this case the network is evaluated better, as games tend to be different. Also, in this way we can see how well the network generalize.

- using Dirichlet noise, this repo manages to randomize(to a certain degree) even the games that are generated in self-play, so the games are more unique, and the network tends to learn better.

- the networks used are almost like those in AlphaZero and AlphaGo Zero, with minor tweaks in order to be able to run them on resource-constrained system(systems with limited RAM, GPU memory, computation power, etc.)

- the MCTS is reset after each game, so neither player has any advantage of having a developed tree and moving first. In certain games the second player has an advantage. This advantage combined with a developed tree makes the game easy for the second player.
- Othello game is updated in order to take a draw into account.

- this implementation provides means of tracking the progress of the network through the training. This info is provided as the number of games won,lost or which resulted in a draw in each epoch against Greedy, Random and Alpha-Beta pruning. However, you can turn this feature off.

- this implementation provide complex neural networks that can be used for every game and are capable of learning any game. The project mentioned above uses very small networks which are unsuitable to learn more complex games, thus not being general enough to be used for all games.

# Ways in which this project is different from AlphaZero and AlphaGo Zero:

- HybridAlpha uses symmetries, unlike AlphaZero. AlphaGo Zero also uses symmetries.

- HybridAlpha uses the evaluation phase, unlike ALphaZero. AlphaGo Zero also uses the evaluation phase.

- HybridAlpha has the goal of mastering any 2-player, perfect information, zero-sum game. This goal is similar to AlphaZero. However, AlphaGo Zero is only capable of mastering the game of GO.

- HybridAlpha uses a network very similar to those provided by AlphaZero and AlphaGo Zero. However, due to the constraints of running and training on resource constrained systems, the shapes of the input and output of the network are smaller. Without this constraint, HybridAlpha can't be run on resource constrained systems.

- HybridAlpha is a sequential algorithm, which means that the generation, training and validation phases executes in parallel. This was done in order to be able to use this algorithm on a resource-constrained system. AlphaZero and AlphaGo Zero are heavily parallelized.

It seems that by using symmetries and evaluation phase, HybridAlpha is better compared with a sequential implementation of AlphaZero when running and training on resource constrained system and when AlphaZero has the same input-output shape as HybridAlpha. Without this constraint, AlphaZero can't be run on resource constrained system and i am unable to test the performances of HybridAlpha against AlphaZero.


# How to run the program

In order to pit the network against another network/Greed/Random/Alpha-Beta/Human player you need to run pit.py.

In order to train the network you need to run main.py

MakeGraph.py is the tool used for generating graphs based on the data that is logged during training

you will find a list of parameters that you want to set in each scritpt. Modify the parameters there in order to take effect.

