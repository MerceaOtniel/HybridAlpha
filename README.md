# AlphaZero
HybridAlpha - a mix between AlphaGo Zero and AlphaZero for multiple games

This project has the goal of creating an Hybrid between AlphaZero and AlphaGo Zero, both published by DeepMind.
Moreover, this is an improved and extended implementation of the project which can be found here https://github.com/suragnair/alpha-zero-general

Ways in which this project improves over the repository presented above are:

-better heuristics for testing the networks

-the addition of Alpha-Beta pruning algorithms which for some of the games also takes into account the depth in the search tree to provide a stronger heuristics and to test the network capacity of generalization.

-the games start from random position when pitting the network, in this case the network is evaluated better, as games tend to be different. Also, in this way we can see how well the network generalize.

-using Dirichlet noise, this repo manages to randomize(to a certain degree) even the games that are generated in self-play, so the games are more unique, and the network tends to learn better.

-the networks used are almost like those in AlphaZero and AlphaGo Zero, with minor tweaks in order to be able to run them on resource-constrained system(systems with limited RAM, GPU memory, computation power, etc.)

-the MCTS is reset after each game, so neither player has any advantage of having a developed tree and moving first. In certain games the second player has an advantage. This advantage combined with a developed tree makes the game easy for the second player.
-Othello game is updated in order to take a draw into account.

-this implementation provides means of tracking the progress of the network through the training. This info is provided as the number of games won,lost or which resulted in a draw in each epoch against Greedy, Random and Alpha-Beta pruning. However, you can turn this feature off.

Ways in which this project is different from AlphaZero and AlphaGo Zero:

-HybridAlpha uses symmetries, unlike AlphaZero. AlphaGo Zero also uses symmetries.

-HybridAlpha uses the evaluation phase, unlike ALphaZero. AlphaGo Zero also uses the evaluation phase.

-HybridAlpha has the goal of mastering any 2-player, perfect information, zero-sum game. This goal is similar to AlphaZero. However, AlphaGo Zero is only capable of mastering the game of GO.

It seems that by using symmetries and evaluation phase, HybridAlpha is better compared with a sequential implementation of AlphaZero when running and training on resource constrained systems.
Moreover, this repo also contains a version of HybridAlpha which doesn't use any symmetries.

In order to pit the network against another network/Greed/Random/Alpha-Beta/Human player you need to run pit.py.

In order to train the network you need to run main.py

MakeGraph.py is the tool used for generating graphs based on the data that is logged during training

