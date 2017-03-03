class: center, middle

# keras talk

---

## Introduction (2min)

- hi, i'm micha
- deep learning has been getting a lot of hype lately
    - it seems it can solve every problem out there!
- but so far it seems secluded to academics and experts
- the waves of pre-trained models and example codes should be an indication to
  the contrary
  - we can easily go and play with many of the amazing models that the papers
    talk about
- deep learning though is easy to get into, hard to master... but mastery comes
  with experimentation
- i want to give you the tools to start that mastery

---

## Overview of Deep Learning (15min)

- Supervised machine learning
    - We have training data that shows us samples and what the output should be
    - We want to learn some method of predicting those outputs for new inputs
- Deep learning is really just a fancy way of composing functions
    - Show function for dense network
        ```
        def feed_forward(input, weights, biases, activation):
            return activation(input @ weights + biases)
        ```
    - Show function for CNN
        ```
        import numpy as np
        from scipy.signal import convolve

        def convnet(input, filters, biases, activation):
            return activation(
                np.stack([convolve(input, f)
                          for f in filter])
                + biases
            )
        ```
    - Show function for RNN
        ```
        def RNN(input_sequence, W, U, biases, activation):
            output = None
            for input in input_sequence:
                output = activation(x @ W + output @ U + biases)
                yield output

        def GRU(input_sequence, W, U, biases, activation_in, activation_out):
            output = None
            for input in input_sequence:
                z = activation_in(W[0] @ input + U[0] @ output + biases[0])
                r = activation_in(W[1] @ input + U[1] @ output + biases[1])
                last_output = z * output +
                              (1 - z) * activation_out(W[2] @ x +
                                                       U[2] @ (r @ last_input) + 
                                                       biases[2])
                yield output
            
        ```
- Why compose functions that way?
    - SGD!
        - http://sebastianruder.com/content/images/2016/09/contours_evaluation_optimizers.gif
        - https://theclevermachine.files.wordpress.com/2014/09/2layer-net-ring.gif
    - Has the advantage of being differentiable
- Other advantages
    - Builds AST... very very portable
    - Think of LLVM
        - different tensor libraries are llvm (theano,  tensorflow, etc)
        - different frameworks are languages (keras, tflearn, nolearn)

---

## Problem Spec (10min)

- There is a lot of text data on the internet
- The amount that each us is expected to read is steadily growing
    - whether it's the news
    - or the long list of documents that come out of a business
- Create meaningful summaries of text!
- Has been done in the past, but with very hack-ish methods
    - show luhn slide
- Can we do better?

---

## Data (2min)

- 


---

## Things I messed up on

- START SIMPLE... easier said than done though
- Make sure your train/valid/test splits don't have any bleeding

## Productionalizing (time permitting)

- float16?
- GPU?
- ...
