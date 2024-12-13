# Zero-shot and Few-shot learning

**Reference:** 
* [neptune.ai blog](https://neptune.ai/blog/zero-shot-and-few-shot-learning-with-llms)
* Fig. 1.6 Build a large language model from scratch

## Key Concepts:

The goal of zero-shot and few-shot learning is to get an LLM model to perform a new task it was not trained for.

* Chatbots based on LLMs can solve tasks they were not trained to solve either out-of-the-box (zero-shot prompting) or when prompted with a couple of input-output pairs demonstrating how to solve the task (few-shot prompting).
* Zero-shot prompting is applicable for simple tasks or tasks that only require general knowledge. It doesn’t work well for complex tasks that require context or when a very specific output form is needed.
* Few-shot prompting is useful when we need the model to **learn a new concept or when a precise output form is required**. It’s also a natural choice with very limited data (too little to train on) that could help the model to solve a task.
* Neither zero-shot nor few-shot prompting will work for complex multi-step reasoning to yield good performance. In these cases, we need **fine-tuning**.

### Zero-shot learning
* Zero-shot prompting refers to simply asking the model to do something it was not trained to do.
* The word `zero` refers to giving the model **no examples of how this new task should be solved**. We just ask it to do it, and the Large Language Model will use the general understanding of the language and the information it learned during the training to generate the answer.
* For example, we can ask a model to translate a sentence from one language to another and it will likely produce a decent translation, even though it was never explicitly trained for translation. Similarly, most LLMs can tell a negative-sounding sentence from a positively-sounding one without explicitly being trained in sentiment analysis.

### Few-shot learning
* Few-shot prompting refers to **asking a Large Language Model to solve a new task while providing examples** of how the task should be solved.
* This is done by passing a small sample of training data to the model through the query, allowing the model to learn from the user-provided examples. However, unlike during the pre-training or fine-tuning stages, the learning process does not involve updating the model’s weights. Instead, the model stays frozen but uses the provided context when generating its response. This context will typically be retained throughout a conversation, but the model cannot access the newly acquired information later.

<img width="755" alt="image" src="https://github.com/user-attachments/assets/60ffb00b-6f80-4482-8caa-202f27282ac4" />



