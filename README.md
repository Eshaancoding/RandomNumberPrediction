# Random Number Prediction

Using FCN (Fully Connected Layers) to predict the next random number from python's random.randint. As you might expect, it doesn't work.
The reason why is because FCN is not good at picking up dependent data from a dataset. If you think about it, each image from a MNIST dataset, for example, is independent from another image. However, `randint` function is dependent. If you were to get a list of random numbers (assuming that the seed is constant), and get another list of numbers from that same seed, it would depend on the first sequence of numbers. All because the seed is constant. 

My model just picks one number, cause it has given up trying to predict the next number in that sequence. So I wouldn't be surprised if the accuracy of near 0.02 (if you change the `max_random_num` parameter, then the accuracy should be at `1 / max_random_num`.

This project is just a lesson learned for me I guess.    
