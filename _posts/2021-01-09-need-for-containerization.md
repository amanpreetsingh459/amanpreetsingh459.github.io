---
permalink: /need-for-containerization
layout: post
title: 'What is "Containerization" and its role in productionizing ML model?'

---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Content
* Overview (the problem statement)
* Virtualization
* Containerization
* ML lifecycle
* ML in containers
    * Tools to achieve ML in containers (Docker, Kubernetes)
    * Brief about the major cloud providers of container services
* Conclusion
* References

## Overview (the problem statement)

The pace at which the field of Machine Learning or more generally the field of Software Engineering is growing, we come across the ever-changing dependencies problem of the software components. Below are the examples which demonstrate the same: - 

1. Suppose at one time we have an application running in production infrastructure with its underlying programming language version “X”. To leverage the infrastructure as its whole we want to deploy another application on it, but in this case the underlying programming language version is greater than before, say “X.1.1” along with few other libraries which are not relevant to the already running app on the infrastructure. How to rescue this situation so that we could leverage the production infrastructure entirely?

2. As for an ML application which often comprises of the multiple models. Say a language translation application for the text written on an image. One model will be used to extract text from the image, second to detect its language, then the third to do the translation in a target language. Moreover, when there are multiple languages to/from to convert. We can imagine the pairs of language translation models. Is it feasible to get the separate infrastructure for each of these models to have its own isolated dependencies?

Below we will be discussing few methods which address the problems we discussed above.

## Virtualization

Virtualization is a technique to deploy one or more guest operating systems on top of a host operating system. This allows developers to run multiple operating systems (called virtual machines) on just the one physical machine enabling them to cope with the different dependencies of different applications. A thin layer of software called a “hypervisor” decouples the virtual machines from the host and dynamically allocates computing resources to each virtual machine as needed. The advantages of virtualization include: -

* Multiple operating systems can run on the same physical machine
* Maintenance and Recovery becomes easy in case of any failure
* Total cost of ownership also becomes less due to the reduced need for infrastructure

<br>
<div class="imgcap">
<img src="/assets/images/blog6_need-for-containerization/blog6_virtualization.jpg">
</div>
<br>

Running multiple Virtual Machines in the same host operating system leads to performance degradation. Because of the guest OS running on top of the host OS, has its own kernel and set of libraries and dependencies. This takes up a large chunk of system resources, i.e. hard disk, processor and especially RAM. So along with the above advantages, virtualization comes with few limitations as well. They are: -

* Running multiple Virtual Machines leads to unstable performance
* Hypervisors are not as efficient as the host operating system
* Boot up process is long and takes time

## Containerization

A container is basically a standardized unit of software. Containerization is the technique of accomplishing virtualization at the operating system level. Whereas Virtualization brings abstraction at the hardware level, Containerization brings abstraction at the operating system level. Containerization is the Virtualization itself. Containerization is however more efficient because there is no guest OS here and utilizes a host’s operating system, share relevant libraries & resources as and when needed unlike virtual machines. Application specific binaries and libraries of containers run on the host kernel, which makes processing and execution very fast. Booting-up a container takes only a fraction of a second. Because all the containers share, host operating system and holds only the application related binaries & libraries. They are lightweight and faster than Virtual Machines.

Advantages of Containerization over Virtualization: -

* Containers on the same OS kernel are lighter and smaller
* Better resource utilization compared to VMs
* Boot-up process is short and takes few seconds

<br>
<div class="imgcap">
<img src="/assets/images/blog6_need-for-containerization/blog6_containerization.jpg">
</div>
<br>

Containers are an abstraction at the app layer that packages code and dependencies together. Multiple containers can run on the same machine and share the OS kernel with other containers, each running as isolated processes in user space. Containers take up less space than VMs (container images are typically tens of MBs in size), can handle more applications and require fewer VMs and Operating systems.

## ML lifecycle

The Machine Learning lifecycle has been defined as cyclic process which majorly involves 3 pipelining steps. We will briefly touch each of these: - 

**1. Data pipeline: -** A typical machine learning project starts with acquiring the data in its raw form. Then the data is cleaned. Cleaned dataset is often used repeatedly. If the desired cleaned form of data is not achieved the data acquisition process starts again. Then the data is transformed into a form that can be feed into an ML algorithm. This process is often called vectorization.

**2.    Training pipeline: -** This stage typically concerned with training the model from the data (vectorized) given to it. The training is basically an optimization process to reduce the loss (A metric to measure ML model’s performance). During this process appropriate model parameters such as coefficients of the model’s mathematical equation are found. Usually a part of the training data is saved to measure the model’s overall performance on the new data. If the results are not satisfactory, the training process is performed again ang again with different configuration of the model.

**3.    Inference pipeline: -** When the model’s performance on the test data is satisfactory, it then deployed as part of any app on the web or mobile devices. One goal of this phase is that model behaves with its proper functionality and secondly it keeps on recording various metrics which would require in retraining the model after a period. This may be required due to the potential changes in the nature of dataset. Or periodic improvements are needed in model.

The typical smaller steps comprising in all the 3 phases discussed above are illustrated below in the chart: -

 
<br>
<div class="imgcap">
<img src="/assets/images/blog6_need-for-containerization/blog6_ml_lifecycle.png">
</div>
<br>

## ML in containers

As from the previous section of ML lifecycle we have seen that there are lot of steps in ML process. Each of these steps require separate toolkit in form of libraries with their specific versions, operating environments, always varying infrastructure (Storage, RAM) requirements. 

When there are so much dynamic needs to just a single application, its isolation is quite the challenge. Because the production systems are the shared pools of resources which different applications distribute among themselves. A bundled solution is needed that can contain all the application specific dependencies, in a version-controlled manner and which is self-contained, isolated from the other apps on the same infrastructure. Of-course containers are the solution.

The benefits of containers which we discussed above in the section “Containerization”, makes them perfect fit for an ML application.

### Tools to achieve ML in containers (Docker, Kubernetes)

#### Docker

Docker is a containerization platform that packages an application and all its dependencies together in the form of a docker container to ensure that the applications works seamlessly in any environment. Containers are built in form of a docker image, which is an immutable(unchangeable) file that contains source code, libraries, dependencies, tools and any other files needed to run an application. Due to their read-only quality, these images are sometimes referred to as snapshots. They represent an application and its virtual environment at a specific point in time. This consistency is one of the great features of Docker. It allows developers to test and experiment software in stable, uniform conditions.

It all starts with a script of instructions that define how to build a specific Docker image. This script is called a Dockerfile. The file automatically executes the outlined commands and creates a Docker image. The image is then used as a template (or base), which a developer can copy and use it to run an application. The application needs an isolated environment in which to run – a container.

#### Kubernetes

Kubernetes is an open-source orchestration software that provides an API to control how and where the containers will run. It allows to run Docker containers and workloads and helps with tackling some of the operating complexities when moving to scale multiple containers, deployed across multiple servers. Containers are grouped into pods, the basic operational unit for Kubernetes. These containers and pods can be scaled to any desired state and provide the ability to manage container lifecycle.

While the promise of containers is to code once and run anywhere, Kubernetes provides the potential to orchestrate and manage all the container resources from a single control plane. It helps with networking, load-balancing, security and scaling across all Kubernetes nodes which runs containers. Kubernetes also has built-in isolation mechanism like namespaces which allows the grouping of container resources by access permission, staging environments and more. These constructs make it easier for IT to provide developers with self-service resource access and developers to collaborate on even the most complex microservices architecture without mocking up the entire application in their development environment. Combining DevOps practices with containers and Kubernetes further enables a baseline of microservices architecture that promotes fast delivery and scalable orchestration of cloud-native applications.

**Using Kubernetes with Docker will: -**

1. Make the infrastructure more robust and app highly available. App will remain online, even if some of the nodes go offline.
2. Make the application more scalable. If an app starts to get a lot more load and there is a need to scale out to be able to provide a better user experience, it is simple to spin up more containers or add more nodes to the Kubernetes cluster.

 

### Brief about the major cloud providers of container services
Below are the 3 major cloud provides list which provide the ML in containerization fashion as-a-service: -

* <a href="https://cloud.google.com/ai-platform">**https://cloud.google.com/ai-platform**</a>
* <a href="https://azure.microsoft.com/en-us/services/machine-learning">**https://azure.microsoft.com/en-us/services/machine-learning**</a>
* <a href="https://aws.amazon.com/sagemaker">**https://aws.amazon.com/sagemaker**</a>


## Conclusion

Essentially, containers are lightweight, stand-alone components containing applications using shared operating systems as opposed to virtual machines, that require emulated virtual hardware. Docker enables us to easily pack and ship applications as small, portable and self-sufficient containers, that can virtually run anywhere. Kubernetes provides orchestration facility for the containers, enabling them to scale at any level we desire. Within whole scenario ML fits best as a use case to make use of containers up to their full potential.

 

Hope you enjoyed the post...


## References
* <a href="https://www.edureka.co/blog/docker-explained/">https://www.edureka.co/blog/docker-explained/</a>
* <a href="https://azure.microsoft.com/en-in/topic/kubernetes-vs-docker/#:~:text=A%20fundamental%20difference%20between%20Kubernetes,production%20in%20an%20efficient%20manner.">Azure guide - Kubernetes vs. Docker</a>
* <a href="https://www.bmc.com/blogs/machine-learning-containers/">https://www.bmc.com/blogs/machine-learning-containers/</a>
* <a href="https://github.com/tm1611/Machine-Learning-Engineer-Nanodegree">https://github.com/tm1611/Machine-Learning-Engineer-Nanodegree</a>
* <a href="https://www.educba.com/machine-learning-life-cycle/">https://www.educba.com/machine-learning-life-cycle/</a>
* <a href="https://fntlnz.wtf/post/why-containers/">https://fntlnz.wtf/post/why-containers/</a>

