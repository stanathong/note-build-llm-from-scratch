# Appendix E: Parameter-efficient finetuning with LoRA

## E.1 Introduction to LoRA

* LoRA is a technique used for parameter-effient fine-tuning.
* It adjusts only a small subset of the model's weight parameters.
* The "low-rank" **limits** model adjustments to a smaller dimensional subspace of the total weight parameter space.
* By doing so it captures the most influential directions of the weight parameter changes during training.
* When training deep neural networks, during backpropagation, we learn $\Delta W$ matrix to update the original model's weight parameters.
* In regular training and fine-tuning, the weight update is defined as follows:

     $W_{updated} = W + \Delta W$

* LoRA instead computes the weight updates $\Delta W$ by learning an approximation of it:

     $\Delta W  \approx AB$
  
* where $A$ and $B$ are two matrices smaller than $W$, and $AB$ represents the matrix multiplication product between $A$ and $B$.
* In LoRA, we can reformulate the weight update as

     $W_{updated} = W + AB$


<img width="703" alt="image" src="https://github.com/user-attachments/assets/6c705572-f42f-4626-ad2c-9cbeca9a827a" />

* In case of regular fine-tuning with x as the input data:

     $x(W + \Delta W) = xW + x \Delta W$

* In LoRA,

     $x(W + AB) = xW + xAB$

* The ability to keep the LoRA weight matrices $AB$ separates from the original model weights $W$
allows for the pretrained model weights to remain unchanged and the LoRA matrices being applied
dynamcially after training when using the model.

* Keeping the LoRA weights separate is useful because we don't need to store multiple complete versions
of an LLM, as only the smaller LoRA matrices are needed.

## E.4 Parameter efficient fine-tuning with LoRA

* To modify and fine-tune the LLM using LoRA, we initialise
    * a LoRA Layer that creates the matrices A and B
    * the alpha scaling factor: it indicates the degree to which the output from LoRA can affect the original layer's output.
    * the rank r setting: it governs the inner dimensions of matrices A and B 

<img width="607" alt="image" src="https://github.com/user-attachments/assets/29956712-87f7-4a69-ac68-a9e5cccebb4e" />
