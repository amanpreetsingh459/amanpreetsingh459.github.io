---
layout: post
title: Machine Learning in Research vs Production

---

# Content

* Overview (the problem statement)
* Factors differentiating the both
    1. Data
    2. Users/Consumers of models
    3. Hardware priorities
    4. Cost
    5. Explainability/Interpretability
    6. Timeliness of Development/Deployment
    7. ML Model vs ML Product
* References



## Overview (the problem statement)
The use of machine learning applications in the industry is in its very early stage (not sure we will ever be able to call it 'past very early stage'). Until very recently most of the Machine Learning was happening in the research or academic institutes only where the scholars/researchers were working to build the models which could be really useful for the masses. Then when there was notable success in making useful mathematical(Machine Learning) models, the next challenge was to make the model available for the public to use. That is to use these fine models in production.

In general, the main difference between doing machine learning in research and in production is the focus. Machine learning research is focused on developing new algorithms and techniques that can improve the state-of-the-art in the field. This often involves experimenting with different approaches, testing their performance on a variety of tasks, and comparing their results to existing methods. In contrast, machine learning in production is focused on deploying machine learning models in a real-world setting and using them to solve specific problems. This often involves working with a team of engineers and data scientists to design, implement, and maintain machine learning systems that are reliable, efficient, and effective. Usually in production the goal is to create a useful product which would be used by either a specific set of people or by everyone. The product can be of either for-profit or non-profit.

So as of today one must have a good idea of both aspects of Machine Learning. There are a lot of factors which differentiates the both. We will discuss those in this blog-post.



## Factors differentiating the both

### 1. Data (Requirements, nature(static, constantly shifting))
Data is the fuel to any Machine Learning algorithm no matter if it's in research or production. While in research we often work with the datasets which are publically available and mostly are cleaned and well formatted. The major focus remains on developing the model which would be proving some hypothesis right or wrong. Using the pre-cleaned data frees up the researchers from spending a lot of time and effort to acquire it, clean it, remove bias, (labelling) etc. It allows us to spend the entire time/effort to develop and train the models. There are a lot of popular datasets which are used for setting up benchmarks such as [SQUAD](https://huggingface.co/datasets/squad), [IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), [GLUE](https://huggingface.co/datasets/glue) to name a few.

In production on the other hand, data is not actually available first of all. We have to acquire it through either web scraping or crowd-sourcing. Or we might need to work with user data which may come up with privacy and regulatory concerns. Unlike publically available datasets this data is not very clean and ready-to-use. It's messy, unstructured, biased, incomplete, (not-labelled) etc. Moreover, it's constantly changing over time. So it's quite challenging and there goes a lot of effort in making it ready to be fed into the model for training and development. And unlike already available historical datasets, the production datasets are always dynamic in nature, only one kind of preprocessing cannot be universal for a production dataset. So the preprocessing/cleaning methods also tend to be changed over time as per the available dataset.

Below is a brief table depicting the major differences between research and prod datasets:-


|     | Data in Research                  | Data in Production                   |
| --- | ---                               | ---                                  |
| 1.  | Available                         | Not always available, need to acquire |
| 2.  | Historical                        | Historical and Current               |
| 3.  | Static                            | Dynamic, Constantly changing         |
| 4.  | Clean                             | Messy                                |
| 5.  | Anonymised or no privacy concerns | Privacy and regulatory concerns      |



### 2. Users/Consumers of model
The consumers of machine learning in research and production can be different in several ways. In research, the consumers of machine learning models are typically other researchers and academic institutions who are interested in using the latest algorithms and techniques to advance the field. These consumers may be interested in using machine learning to solve complex problems, test new theories, or develop new applications.

In production, the consumers of machine learning are typically businesses, organizations, or individuals who are looking to use machine learning to solve specific problems or improve their operations. These consumers may be interested in using machine learning to automate processes, make predictions, or improve decision making.

Usually when a model is built in research, their consumers are not obvious at the time of development. It largely depends on the validity of the hypothesis on which a research project is being executed. The project might or might not get successful. The success of the model determines the consumer set for the model. And ofcourse with the success of a machine learning model there can evolve a number of applications with the help of the model, thus the consumers. So initially the focus in research largely remains on building a useful ML model first.

In contrast, in research the model development is specifically targeted on a specific set of consumers. The purposes can be many from making any prediction to automating something. The number of consumers so that the scale of the model could be determined.

The major difference if we see from the consumers from the research could be the dynamic nature of the consumers. In prod the consumer's set tends to be changed after a certain time period.





### 3. Hardware priorities
**Hardware priorities in research:**

In research, hardware requirements may be driven by the size and complexity of the data sets being used, as well as the computational demands of the machine learning algorithms being used. Researchers may prioritize high-performance CPUs, GPUs, or specialized hardware such as Tensor Processing Units (TPUs) to accelerate the training process. The inference part is not really(figures crossed) considered till the model building is done and a starting point of the hypothesis has been concluded to be useful.

**Hardware priorities in production:**

In production, hardware requirements may be driven by the volume of data and requests being processed, as well as the reliability and scalability of the system. In production specialized hardware could be used that is able to handle large volumes of data and requests, as well as hardware that is able to tolerate failures or outages without disrupting service. Because the major focus is on serving client requests that may vary from a few per minute to 100s or more per second. Additionally, the production machine learning projects may require different hardware and software tools that can support the deployment, monitoring, and maintenance of machine learning systems at scale. This might include distributed systems, specialized hardware, and software tools for deployment and management.



### 4. Cost

Another key difference between machine learning in research and production is the cost considerations.
Research projects may involve higher costs due to the need for powerful hardware and specialized software, as well as the cost of conducting experiments. Usually the cost in research usually comes from the excessive use of specialized GPUs or TPUs and that too on an experimental basis. Their hyperparameter tuning involves a lot of trying and testing of different combinations which usually are the main reasons for increased cost of machine learning projects in research.

In contrast, production machine learning projects may involve different cost considerations, such as the cost of deploying and maintaining machine learning systems at scale, the cost of integrating machine learning into existing systems and processes, and the cost of training and managing a team of data scientists and engineers. It entirely depends upon the scale on which the model will be used by different consumers.

Overall we cannot conclude in which scenario the model building cost will be less or more. It entirely depends upon the focus of the machine learning model and its intended audience and scale.



### 5. Explainability/Interpretability
The model explainability considerations for using machine learning in research and production can vary depending on the specific context and goals of the project. In general, however, research projects may prioritize model explainability in order to better understand how machine learning algorithms work and to develop new methods for interpreting and explaining their decisions. This might involve using techniques such as feature importance analysis, sensitivity analysis, or model dissection to understand the behavior of machine learning models.

In contrast, production machine learning projects may prioritize model explainability for different reasons. For example, explainable models can help organizations make more informed decisions, comply with regulations, or build trust with their customers. In these cases, explainability techniques may be used to provide insight into how a model is making predictions, and to identify potential biases or limitations in the model.



### 6. Timeliness of Development/Deployment
In general, however, research projects may have longer timelines for development due to the need for experimentation, testing, and validation of new algorithms and techniques. This can involve conducting multiple rounds of experiments, comparing the performance of different algorithms, and refining the approach over time.

In contrast, production machine learning projects may have shorter timelines for development due to the need to deploy machine learning models and systems quickly and efficiently. This can involve working with a team of data scientists and engineers to design, implement, and deploy machine learning systems in a production environment. In these cases, timeliness is often a critical factor, as organizations may need to deploy machine learning models quickly in order to stay competitive and achieve their goals.



### 7. ML Model vs ML Product
The one sentence difference is that in research the focus is on building/creating a model, and in production the focus is on building a product that uses a machine learning model, one or several.

A machine learning model is a mathematical model that is trained to perform a specific task, such as predicting a target variable given a set of input features. The model is typically trained on a large dataset and uses a specific machine learning algorithm to learn the underlying relationships in the data. The trained model can then be used to make predictions on new data.

On the other hand, a machine learning product is a complete system that incorporates machine learning models and other components to perform a specific task or solve a specific problem. A machine learning product may include a user interface, a database, and other backend systems in addition to the machine learning models. It is designed to be used by end users and may be deployed in a production environment where it can be accessed and used by multiple people.







## References
* https://huggingface.co/datasets/squad
* https://huggingface.co/datasets/glue
* https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
* https://syncedreview.com/2019/06/27/the-staggering-cost-of-training-sota-ai-models/

