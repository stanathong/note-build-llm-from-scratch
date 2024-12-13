# Zero-shot and Few-shot learning

**Reference:** 
* [neptune.ai blog](https://neptune.ai/blog/zero-shot-and-few-shot-learning-with-llms)

## Key Concepts:
* Chatbots based on LLMs can solve tasks they were not trained to solve either out-of-the-box (zero-shot prompting) or when prompted with a couple of input-output pairs demonstrating how to solve the task (few-shot prompting).
* Zero-shot prompting is applicable for simple tasks or tasks that only require general knowledge. It doesn’t work well for complex tasks that require context or when a very specific output form is needed.
* Few-shot prompting is useful when we need the model to **learn a new concept or when a precise output form is required**. It’s also a natural choice with very limited data (too little to train on) that could help the model to solve a task.
* Neither zero-shot nor few-shot prompting will work for complex multi-step reasoning to yield good performance. In these cases, we need **fine-tuning**.

