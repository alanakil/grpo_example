# Group Relative Policy Optimization (GRPO)

Recently, DeepSeek took the world by storm when they presented DeepSeek-R1 and showed the world how to train reasoning models.
Shortly after, I went through the technical details of the papers (notes coming soon in markdown), so I decided
to implement this form of Reinforcement Learning (RL) in a smaller language model that I could train in my home PC.

Therefore, I finetuned a Qwen-2.5-0.5B-base model on the GSM8K dataset, which is a dataset of grade school math problems. 
I show here how the model can go to not being able to answer any question in the right format, to reaching ~23% accuracy on the benchmark after RL-training on 1,868 examples.
Additionally, I show how not all training steps are necessary to reach this performance, in fact the model starts to understand the task well after 400 examples when its accuracy jumps from 0% to ~10%.

This is a work in progress and I have a few things coming up that I will add soon:
- Specific examples of how it responds to questions throughout the training process. When does reasoning emerge?
- Experiment with other math datasets
- Tests on other small models like Llama-3.2-1B-base
- How does the structure of such a model differ from its -Instruct version?
