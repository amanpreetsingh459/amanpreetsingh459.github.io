---
layout: post
title: 'When "Not" to Use Machine Learning?'

---

# Content

* Overview (the problem statement)
* Challenges associated with a Machine Learning project
    * Data
    * Cost of training an ML model
    * Interpretability of results
    * Ethics and responsibility
* Breakdown of the limitations into guidelines to not to use ML
* Endnotes
* References

## Overview (the problem statement)

> ***"You haven't mastered a tool until you understand when it should not be used." – <a href="https://twitter.com/kelseyhightower/status/963428093292457984?s=20">Kelsey Hightower</a>***


When we try to solve a problem, in what circumstances should we apply machine learning? Is it true that in every circumstance, machine learning will always outperform rule-based and/or heuristic approaches? The answer to both the questions is NO.! Though Machine Learning systems have proven really useful in many real-world use cases, be it understanding the natural language or understanding digital images. But it should not be used only because its trendy at present time or its cool to use ML. There are various factors that impact the success of an ML project. The very first thing is to have a clear business goal. Then measurement of the returns on investing (ROI) into the resources required to execute an ML project etc.

Overall in this blog-post we will discuss the challenges associated with executing ML projects and scenarios where it's not feasible to use ML. Let's jump right in..

## Challenges associated with a Machine Learning project

### Data

The data is the basic fuel for the success of any machine learning based solution. Below are the few difficulties related with the topic "data" while doing Machine Learning:-

* **Amount or size of dataset:** Machine learning algorithms require large amounts of data before they begin to give useful results. Till date neural networks based ML algorithms have proven to be giving the best results. And they require copious amounts of training data (and so the processing power). The larger the architecture, the more data is needed to produce viable results. Reusing data is not a good idea, and data augmentation is useful to some extent only, but having more data is always the preferred solution.

* **Quality of data:** The data should be of good quality. It is of good quality if it does not have any unrelated, ambigous or wrong information about the problem at hand. With anyone of these issues, an ML algorithm cannot produce the results which are good enough to produce the value. The valuable results are often those which improve over human-level error. 

* **Labeling of data:** In case of supervised machine learning, which is by far the most successful type of ML, The data should be labled. That is every instance of data should have its true label associated with it. Manual labeling of the data is very expensive. Though there are methods to handle unlabled data too, that is unsuepervised machine learning, but they have very limited applicability when we talk about real world value provided applications. There is still so much work to do. This is an active area of reasearch and hopefully in near future we might have good set of methods which can produce good results from unlabled data too.

* **Skewed or Biased or unbalanced data:** This issue is seen more often in the real world. This problem occurs when the total number instances of one specific category is way more(or less) than the other category in the dataset of the problem at hand. This creates serious issues for an ML model as, the model may be biased towards the category which was having more number of examples in the dataset. Although there are methods which can be used to deal with this problem to an extent, but if the data is balanced, the full potential of an ML algorithm can be used to create value.

### Cost of training an ML model

The cost is another big challenge in training an ML model. As in the previous section we have seen that an ML algorithm would require massive amount of data, so the cost of training it increases proportionally. Because an ML model can take from few hours to few weeks to train based on the size of data it is being trained on. It's not possible for a common PC to train a really large ML model. 

The ML model involves complex calculation of matrix operations while training. It often requires specialized hardware designed specifically for this task. GPUs, TPUs are few such examples. They come with thier associated costs. There are cloud providers (AWS, Azure, GCP) which provide ML environments as a service, to train and operationalize the ML models.

Another reason of this increasing cost of training an ML model is that the models's usefulness remains only for a limited period of time. This happnes due to the new data which generates overtime for the same problem. With the new data model has to be retrained periodically. This is costly. And this cannot be avoided as well. Because the new data is being generated endlessly and at an exponential rate. So the model also should be retraining overtime.

There are few popular open-sourced ML models like Google BERT (October 2019) and OpenAI GPT-3 (MAY 2020). Due to an estimate BERT model's training cost was USD 6912 and for training GPT-3 it was USD 4.6 million. Here we can estimate the increasing cost of training a value generating ML model in less than 2 years.

### Interpretability of results

Another major challenge is the ability to accurately interpret results generated by the ML algorithms. Interpretability refers to the ability to understand why a model reached to a particular conclusion. There are few ML algorithms like "Linear Regression" and "Random Forest" which are relatively easy to interpret. The results given by them can be travesed back to the question. Neural Networks or deep learning algorithms, on the other hand are difficult to interpret even by the experts. It's often very less likely to know the reasoning behind model giving a particular result.

This is a problem which makes people skeptical about making use of ML models. Though the efforts have been made towards addressing this problem till some extent. There is a cloud service called "Explainable AI" which dedicates for this specific issue related with machine learning.


### Ethics and responsibility

The idea of trusting data and algorithms more than our own judgment has its pros and cons. Obviously, we benefit from these algorithms, otherwise, we wouldn’t be using them in the first place. These algorithms allow us to automate processes by making informed judgments using available data. Sometimes, however, this means replacing someone’s job with an algorithm, which comes with ethical ramifications. Additionally, who do we blame if something goes wrong? The most commonly discussed use-case currently is self-driving cars. Who is to blame if a self-driving car kills someone on the road? Would we trust a machine learning model giving diagnosis about cancer, or we would trust a human doctor instead? This is an active area of debate between machine learning practitioneres and ethical institutions. 


## Breakdown of the limitations into guidelines to 'when not to use ML'

We should not be using Machine Learning when:-

1. We can solve the problem using traditional software development at a lower cost
2. Acquiring accurate data is too complex or impossible
3. Requires large amount of manually labelled data
4. The model needs to go to the market very soon
5. The cost of an error made by the system is too high
6. Performance of the model cannot be audited or guaranteed
7. Each narrow application needs to be specifically trained
8. The model will not have to be improved frequently over time, lacks incremental learning
9. Every decision made by model has to be explainable

## Endnotes

Besides all the limitations and the guidelines to when not to use ML, there are lots of reasons to use ML as well. ML has revolutionalized the world as we know it in the past decade. So from my perspective we should leverage this cool technology which we are witnessing in our lifetime. Hopefully there will be more surprisingly astonishing inventions will be coming with the help of ML. So let's **"Hope for the best and Prepare for the worst"**. 



Hope you enjoyed the post...

## References

* <a href="https://twitter.com/kelseyhightower/status/963428093292457984?s=20">https://twitter.com/kelseyhightower/status/963428093292457984?s=20</a>
* <a href="http://themlbook.com/">http://themlbook.com/</a>
* <a href="https://towardsdatascience.com/when-not-to-use-machine-learning-14ec62daacd7">https://towardsdatascience.com/when-not-to-use-machine-learning-14ec62daacd7</a>
* <a href="https://towardsdatascience.com/the-limitations-of-machine-learning-a00e0c3040c6">https://towardsdatascience.com/the-limitations-of-machine-learning-a00e0c3040c6</a>
* <a href="https://www.onlydeadfish.co.uk/only_dead_fish/2017/01/the-limitations-of-machine-learning.html">https://www.onlydeadfish.co.uk/only_dead_fish/2017/01/the-limitations-of-machine-learning.html</a>
* <a href="https://syncedreview.com/2019/06/27/the-staggering-cost-of-training-sota-ai-models/">https://syncedreview.com/2019/06/27/the-staggering-cost-of-training-sota-ai-models/</a>
* <a href="https://bdtechtalks.com/2020/08/17/openai-gpt-3-commercial-ai/">https://bdtechtalks.com/2020/08/17/openai-gpt-3-commercial-ai/</a>
