
# Temporal Difference (0) Learning to Play 2048
An agent plays 2048 with a weightless neural network trained by online TD(0) learning. It is a side project aiming to apply my knowledge about reinforcement learning to my favorite casual game - 2048. 

# Demo
Watch demo video [[here]](https://youtu.be/y_ntwcheB78)

# How to train and play
You can modify hyperparameters in `globalconfig.py` and train the model by running the `train.py`. To see the model play, just run `play.py`.
You can also easily upgrade the algorithm (e.g. TD lambda, off-policy Monte Carlo,...) as the code is well modulized and readable.

# The "deep" version
Formerly, I made the agent by a convolutional neural network. It worked. It quickly reached 1024. But, to go further, it needs to be trained thoroughly. With the limited resource, I could not go with it till the end. You can see the code in branch `dev`.

# The "optimized" version

# Reference
M. Szubert and W. Jaśkowski, “Temporal difference learning of N tuple networks for the game 2048,” in Proc. 2014 IEEE Conf. Comput. Intell. Games, Dortmund, Germany, 2014, pp. 1–8. DOI: 10.1109/CIG.2014.6932907.
