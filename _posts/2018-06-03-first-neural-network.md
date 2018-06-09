---
layout: post
title: Creating learning(intelligence) into machines using neural networks
---

## How do the humans learn - The concept of intelligence

We are naturally intelligent. Nature has given as the brain. Our brain is capable of synthesizing a diverse set of inputs we call five senses, and from these we create a hierarchy of concepts. While interacting with our environment we can feel our surroundings, see our obstacles and try to predict the next set of actions. We may try several times and fail but that's fine. Through the process of trial and error, we can learn anything. Such capable our mind is. We as the humans are privileged to have this capability as no other thing(machines) or creature has. This is what makes us the us. Everything we ever felt or experienced, all our thoughts and memories are produced by the brain.

Let's talk something about the fancy science of the human brain- A nerve cell, or neuron, is a cell that receives information from other nerve cells or from the sensory organs and then projects that information to other nerve cells, while still other neurons project it back to the parts of the body that interact with the environment, such as the muscles. Nerve cells are equipped with a cell body, a sort of metabolic heart and an enormous treelike structure called the dendritic field, which is the input side of the neuron. Information comes into the cell from projections called axons. Most of the excitatory information comes into the cell from the dendritic field, often through tiny dendritic projections called spines. The junctions through which information passes from one neuron to another are called synapses, which can be excitatory or inhibitory in nature. The neuron integrates the information it receives from all of its synapses and this determines its output.

## Exhibiting learning into the machines - Artificial Intelligence    

So the rules that govern the brain give rise to intelligence. When we want to exhibit this intelligence into machines we call it Artificial Intelligence as we are creating it by ourselves (say by hand). The techniques which we use to achieve this process of artificial intelligence are called machine learning (algorithms). And one type of machine learning which uses the process of neurons inspired from human mind is so called deep learning.

(We are starting using the term algorithm)
It's the same algorithm that sort of explains the functionality of human mind. We can count the major inventions which have helped human race to survive and become better than anything else on the planet earth. The things like modern language to communicate with humans, foot-stepping on the moon etc. But as humans are also the species like many other we also face the threat of our non-existence after a certain period (few million years). Climate change, an asteroid impact or a biological warfare are some of the example which may be threatening to humans' existence.

We may be not too far to solve these problems as of today. But the knowledge level what we possess today has come from the generations of our biological neural networks developments.

So suppose if at this point of time we face one of the threats mentioned above and human species comes to its extinction level. Then whatever or whoever remains, the people or creatures, what amount of knowledge they would have left with comparing with what we have today so that they could create the life again? Not very!

But what if we could harness this already possessed knowledge, what if we could store it in some way and find a way too to access it by the upcoming generations or species. The idea to make  an artificial brain and have it run on a non-biological substrate like silicon(why silicon? because modern computers which are the great storage systems run on silicon based systems itself!). We can give it tremendous computing power and lots of data too and have it solve the problems thousands or millions times faster than a human would solve it alone after learning from scratch.

## Something about history of neural networks

In 1943, two computer scientists named Warren McCulloch and Walter Pitts invented the first computational model of a neuron. That model demonstrated a neuron that received binary input, summed them, and if the sum exceeds a certain threshold value, output a 1 other wise output a 0. So that was the very first attempt when the things about storing the intelligence started. The field is maturing since those days. So we are not talking anything new here.
    
Then it was felt by the later scientists that their model however represents the good computational model but still lacks mechanism for learning.
    
## Perceptron

So a new idea was invented based on the same computational model. They called it the 'perceptron'. The perceptron is basically a 'single layer feed-forward neural network'. We call it feed 'forward' because the data flows only in one direction-forward. The perceptron incorporated the idea of weights on the inputs. So data which is going to be inputted into the neural network is having some weights included with them. The weight basically specifies the importance of particular input parameter. These weights are mathematically applied to the inputs such that after each iteration the output gets more accurate than before.

As a whole the thing is to learn a function from the network by increasing or decreasing the weight values iteratively based upon what outputs are getting generated. This iterative process will be repeated until the output gives satisfactory results. 

This process of iteratively chainging the weights based upon its output values is called the learning process.

## Building our own artificial neural network

The stage is set. We are ready to create our own single layer neural network. The prerequisites are only the knowledge of basic python syntax.

```python
import numpy as np
class NeuralNetwork():
    def __init__(self):
        np.random.seed(3)
        self.weights = np.random.random((3,1))
        
    def __sigmoid(self, number):
        return 1/(1+np.exp(-number))
    
    def __sigmoid_derivative(self, number):
        return number * (1 - number)
        
    def train(self, train_inputs, train_outputs, n_epochs):
        count = 0
        for iteration in range(n_epochs):
            count += 1
            output = self.calculate_logits(train_inputs)            
            error = train_outputs - output
            output_derivative = self.__sigmoid_derivative(output)
            delta = error * output_derivative
            weight_deltas = np.dot(train_inputs.T, delta)
            self.weights += weight_deltas
            if count%10000 == 0:
                print(output)                                        
            
    def calculate_logits(self, train_inputs):
        return self.__sigmoid(np.dot(train_inputs, self.weights))        
```

```python
def main():
    train_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    train_outputs = np.array([[0, 1, 1, 0]]).T
    
    neural_network = NeuralNetwork()

    print("Supplied outputs into the network to learn from: ")
    print(train_outputs)

    print("Machine learnt outputs after training: ")
    neural_network.train(train_inputs, train_outputs, 10000)            
```

```python
if __name__ == "__main__":
    main()

######## Program ends here ##########
######## output ###############

Supplied outputs into the network to learn from: 
[[0]
 [1]
 [1]
 [0]]
Machine learnt outputs after training: 
[[ 0.00966619]
 [ 0.99211858]
 [ 0.99358984]
 [ 0.0078644 ]]
 
We can see above how closely a simple neural network has learnt the outputs from the supplied ones
```

### Endnotes

This idea of using perceptrons to make the neural networks learn was not very popular earlier as there was neither enough data available nor there was enough computing power. But in recent years as we have seen the explosion in data and also in computing power the neural networks have proven to be really great performers to solve the tasks that only humans could do. It made possible by taking massive data and made the neural network several layers deep. An artificial neural network doesn't really exist in physical space. It's an abstract concept we have created programmaticallyand it's represented on silicon transistors. Although they are completely different from the exact functioning of the human brain, they both use very similar approaches for the learning.

Hope you enjoyed the post..
