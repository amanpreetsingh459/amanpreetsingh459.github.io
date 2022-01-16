---
layout: post
title: Practical Guide on ML Model serving using "Tensorflow Serving"'

---

# Content

* Overview (the problem statement)
* Tensorflow serving: The introduction
* Installation
* Prepare a tensorflow model (Build, Train, Save)
* Serving and Consuming the model
* Endnotes
* References

## Overview (the problem statement)
What exactly is the meaning of model-serving. Training a well-performing machine learning model is the first thing, second is to make the model available to the end-users. That involves making our model accessible either through a web page by hosting our model on a web server or through a mobile app by making the model file part of the app. There are 3 major things to consider while deploying the model to be served to the end-users: **the model itself, the model running environment and the input data**. These things are used when inference(call to the model for getting predictions) is made. Whole process is broken down in the following steps:- user supplies the input data and makes a call to the model via rest end-point, end point invokes the model in the model running environment, computes the predictions on the supplied input data and then sends back the predictions to the user.

Though the machine learning models can be served using popular web frameworks like flask or django, they lack some of the major functionalities that are vital for the ML models. They are
1. Missing version control system: Models are often required to be retrained with the new data to keep it relevant. Web frameworks which are mentioned earlier, lack that. Though can be implemented by hand but would be very tricky and difficult to manage
2. Slow model inference: While serving ML models, getting fast response is crucial for the success of the model. By nature the mentioned web frameworks are not designed to server ML models with specific latency requirements

Tensorflow serving solves these issues for us for the good. It has been specifically designed to server ML models by keeping version management, inference with minimum latency, isolated model code management from the other devops code etc. requirements at the top priorities.

## Tensorflow serving: The introduction
TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be easily extended to serve other types of models and data. <a href="https://www.tensorflow.org/tfx/guide/serving">source</a>

In this blog-post we are going to see what are the minimal things(in code) required to serve a tensorflow model with tensorflow serving and docker. Docker is a containerization platform that packages an application and all its dependencies together in the form of a docker container to ensure that the application works seamlessly in any environment. Containers are built in the form of a docker image, which is an immutable(unchangeable) file that contains source code, libraries, dependencies, tools and any other files needed to run an application. More about the docker and containerization can be read in my another blog-post <a href="https://amanpreetsingh459.github.io/2021/01/09/need-for-containerization.html">here</a>. 

<br>
<div class="imgcap">
<img src="/assets/images/blog_tfserving/tf_ext.png">
</div>
<br>

[image_credit](https://www.tensorflow.org/site-assets/images/project-logos/tensorflow-extended-tfx-logo-social.png)

## Installation
We will be using tensorflow serving with docker, which is the easiest way of deploying ML models using tensorflow serving. To install docker go to <a href="https://docs.docker.com/engine/install/">this link</a>. Then install tensorflow serving:
```bash
docker pull tensorflow/serving
```

## Prepare a tensorflow model (Build, Train, Save)
Below we will create a simplest sentiment-classifier model of IMDB reviews. We will use tensorflow for this purpose, a pre-loaded dataset from tensorflow datasets.
```bash
pip install tensorflow==2.5.0
pip install tensorflow_datasets
```

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pickle
import time

#download and load dataset
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

#separate train and test data
train_data = imdb['train']
test_data = imdb['test']

#fetch the tokenizer
tokenizer = info.features['text'].encoder

#test tokenizer
test_string = "That movie i saw the other day was fantabulous"
print("test string: ", test_string)
tokenized_string = tokenizer.encode(test_string)
print("tokenized string: ", tokenized_string)
decoded_string = tokenizer.decode(tokenized_string)
print("decoded string: ", decoded_string)

for token in tokenized_string:
  print('{} ----> {}'.format(token, tokenizer.decode([token])))

# saving the tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#Hyperparameters
BUFFER_SIZE = 10000
BATCH_SIZE = 64
embedding_dim = 64
num_epochs = 10

#shuffle and pad dataset
train_dataset = train_data.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

#prepare the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#print model summary
print(model.summary())

#model compilation
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#model training
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
print("Model has trained")

#test an examples on the trained model
test_review = "That movie i saw the other day was fantastic"
tokenized_string = tokenizer.encode(test_review)
prediction = model.predict(tokenized_string)
prediction[np.argmax(prediction)][0]    #output: 1.0

## save the model into well defined directory
ts = int(time.time())
file_path = f"sentiment_classifier/{ts}/"
model.save(filepath=file_path, save_format='tf')

```

The complete notebook with the outputs for the above code can be found at: <a href="https://github.com/amanpreetsingh459/tensorflow_serving">https://github.com/amanpreetsingh459/tensorflow_serving</a>

## Serving and Consuming the model
Now that the model has been trained and saved in a directory "sentiment_classifier" in our system. We will spin up a docker container and serve the model using tensorflow serving. Only one command(on the terminal) will do the trick:-

### serving
```bash
docker run --rm -p 8501:8501 \
  --mount type=bind,\
source=/complete/path/till/directory/sentiment_classifier,\
target=/models/sentiment_classifier \
  -e MODEL_NAME=sentiment_classifier -t tensorflow/serving
```
The command is a bit complex, I can understand. Let's break it down step by step:-
* docker run: This will run the docker images by spinning up a container, which apparently we are going to do

* `--rm`: This option will delete this container after stopping running it. This is to avoid having to manually delete the container. It is a good practice to delete unused containers. This helps our system to stay clean.

* `-p 8501:8501`: This flag performs an operation known as **"port mapping"**. The container, as well as our local machine, has its own set of ports. In order to access the `port 8501` within the container, we need to **"map"** it to a port on our system. In this case it is mapped to the `port 8501` in our machine. This port is chosen as it is the default port to interact with the model through a `REST API`. If we were using a different protocol such as [`gRPC`](https://grpc.io/) we would require to use `port 8500`.

* `--mount type=bind,source=/complete/path/till/directory/sentiment_classifier,target=/models/sentiment_classifier`: This flag allows us to **mount** a directory in our pc to a directory within the container. `source` is the directory in our local system where the model resides and `target` is the directory in the container

* `-e MODEL_NAME=sentiment_classifier`: Will create the environment variable `MODEL_NAME` and assign to it the value of `sentiment_classifier`. This will be helpful later when we might require to access model-name

* `-t`: Attaches a pseudo-terminal to the container so that we can check what is being printed in the standard streams of the container. This allows us to see the logs printed out by TFS.

After running this command TFS will spin up and host the `sentiment_classifier` model.


### consuming
Now that the model is being served on port 8501, the next task is to define the `REST` endpoint url. That goes like this:
```bash
http://localhost:8501/v1/models/sentiment_classifier:predict
```
Let's break down the parts of the url step-by-step:
* `http://`: Stateless protocol to access any url
* `localhost`: The server address where the model resides. Currently the model is hosted on localhost(in our local machine)
* `8501`: TF serving port name on the server
* `v1`: Refers to the version of TFS used. Currently its v1
* `models/sentiment_classifier`: This specifies the model name and its location in the container
* `:predict`: This has to do with the model signature. It can be one of the predict, classify or regress. Currently it's predict

The simplest way to access this URL by passing the data and getting prediction out of it can be done by using curl command in this format: 
```bash
curl -d '{"instances": [<data values>]}' \
  -X POST <url>
  
curl -d '{"instances": [[444, 27, 131, 284, 1, 108, 414, 18, 1042, 4779, 5872]]}' \
  -X POST http://localhost:8501/v1/models/sentiment_classifier:predict
```
* `-d`: The d stands for data. This is the data that we are going to send to the server for the model to process. Since we are communicating with the model via REST we should provide the data in a JSON format. TFS has the convention that the key of this object should be the string instances and the value should be a list that contains each data point that you want to make inference for.
* `-X`: This flag allows you to specify the desired HTTP method. By default curl uses the GET method but in this case it should be POST.

This will output the following:
```bash
[1.0]
```

The text data has been converted into the numbers using the tokenizer then has been passed into the sentiment-classifier model through url. This gives output a number greater than the value 0.5 so that we would consider it a positive review. If the value were less than 0.5, we would consider it a negative review.

One more feasible and recommended way is to write code that would take the input text, convert it into numbers and pass it through the url to get predictions like below:

```python
import pickle
import json
import requests

# load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
# define the url
url = "http://localhost:8501/v1/models/sentiment_classifier:predict"
    
# function to get the predictions
def make_prediction(instances, url):
   data = json.dumps({"signature_name": "serving_default", "instances": [instances]})
   headers = {"content-type": "application/json"}
   json_response = requests.post(url, data=data, headers=headers)
   predictions = json.loads(json_response.text)['predictions']
   return predictions

# define testing examples, these we can take from user inputs as well
test_reviews = ["That movie i saw the other day was fantabulous", 
                "That movie i saw the other day was aweful"]

predictions = []
for review in test_reviews:
    predictions.append(make_prediction(tokenizer.encode(review), url)[0])
    
# code for interpreting model results into positive or negative reviews
for review, pred in zip(test_reviews, predictions):
    print(review)
    print(pred)
    if pred[0] >= 0.6:
        print(":: Review is Positive ::")
    else:
        print(":: Review is Negative ::")
        
"""below is the model output
That movie i saw the other day was fantabulous
[1.0]
:: Review is Positive ::
That movie i saw the other day was aweful
[0.0689155161]
:: Review is Negative ::
"""
```

## End notes
This tutorial is the minimal requirement for serving a machine learning model through tensorflow serving. There are tons of other concepts that can be covered under tensorflow serving. Though equipped with this initial knowledge we can explore even more complex model serving scenarios. 

Hope you enjoyed the post...

## References
* [https://www.tensorflow.org/tfx/guide/serving](https://www.tensorflow.org/tfx/guide/serving)
* [https://github.com/https-deeplearning-ai/tensorflow-1-public](https://github.com/https-deeplearning-ai/tensorflow-1-public)
* [https://github.com/https-deeplearning-ai/machine-learning-engineering-for-production-public](https://github.com/https-deeplearning-ai/machine-learning-engineering-for-production-public)

