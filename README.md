
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
Python is slow. Moreover, this project used a convolutional neural network at the beginning and switched to a weightless network now. I also try to maximize readability and upgradability, not performance. So, this code will take ages to really win the game.
So, I decided to port this code to [Pascal](https://github.com/ThaiDat/Temporal-Difference-Learning-to-Play-2048-Pascal-Version-) which is a high-performance language. I also used many optimization techniques (bitboard, for example). The agent learned thousand times faster. I kept the structure the same, so that you can switch between the two easily. The output model from the Pascal version can be converted to use in this project smoothly by running `convert.py`.

# Reference
M. Szubert and W. Jaśkowski, “Temporal difference learning of N tuple networks for the game 2048,” in Proc. 2014 IEEE Conf. Comput. Intell. Games, Dortmund, Germany, 2014, pp. 1–8. DOI: 10.1109/CIG.2014.6932907.
