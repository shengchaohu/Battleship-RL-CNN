The RL folder containing the implementation of policy gradient. There is just one single ipython notebook and one trained model. Usage is pretty straightforward. (Jiacheng Sheng)

The probabilistic_approach folder contains implementation of probabilistic ones.

The convnet folder contains implementation of CNN model. I have put the trained model in save_models file. By default when you run main.py, it will continue to train the model based on my pre-trained parameter. If you want to test the model, please uncomment the corresponding line in main.py (on line 131). The sample.py file allows you to generate any number of test samples you want by changing the corresponding number in sample.py file.

Note: The code in convnet folder (except sample.py) is largely adapted from https://github.com/junoon53/battleship, who builds the basic game environment. I tuned the CNN model's hyperparameter and get a better result than the original one. (Shengchao Hu)