# Neural-linear reinforcement learning
In recent years, deep reinforcement learning has seen tremendous successes in several benchmarks like Atari 2600 and OpenAI Gym. Feature extraction is extremely important in reinforcement learning, as it influences the quality of the policy. In recent years, convolutional based architectures combined with Q-learning became prominent to handle high dimensional inputs such as images or video in reinforcement learning. However, these models are computationally expensive and complex. This paper evaluates a theory of using representation learning in reinforcement learning by decoupling the feature extraction from value function approximation. This is achieved using a pre-trained or randomly initialized convolutional architecture to extract linear features from the high dimensional input to perform value function approximation on it. We apply this method to Atari Breakout and compare the effect of different convolutional based feature extractors on the performance of a linear SARSA model. We find that our model has performed well in Atari Breakout while only being trained on CPU.

# Requirements
1. Run `pip install -r requirements.txt` from the project directory to install the requirements.
2. Follow the steps in [here](https://github.com/openai/baselines) to install OpenAI Baselines

# Documentation
1. The starting point of the code is `main.py` in the root directory
2. Code is organised into two packages `utils` and `models`.
   1. `models` package consists of all the code files related to feature extractor models and the linear SARSA models
   2. `utils` package consists of all the code files that are common utilities for the model training/evaluation

**Feature Extractor Models Implemented:** dqn_pretrained, dqn_glorot_normal, dqn_glorot_uniform, dqn_pooling, resnet50_pretrained, resnet50_random_init, resnet101_pretrained, resnet101_random_init, resnet152_pretrained, resnet152_random_init

# Run instructions
1. To `train` the model, run
    ```shell
    python main.py train -s <save_dir> -e <feature_extractor_model_name> -w <weights_directory|Optional> 
    ``` 
    This will train the model and save the plots and weights to the specified save directory    

1. To `evaluate` the model on the test data, run
    ```shell
    python main.py evaluate -s <save_dir> -m <model_weights_path> -e <feature_extractor_model_name> -w <pretrained_weights_path|Optional>   
    ```
   This will evaluate the model and save the gameplay video to save directory and prints the rewards
   