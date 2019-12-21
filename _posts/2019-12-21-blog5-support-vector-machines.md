---
layout: post
title: 'Support Vector Machines? Separate data with largest possible margin!'

---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Definition

**Support Vector Machines**(SVMs for short), in the context of machine learning, are the ways to separate data between two(or more) categories. (The context here used in the **supervised machine learning**, one in which all the data examples are labeled with their correct categories). Basically an SVM model is representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap**(hyperplane)** that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on the side of the hyperplane on which they fall. Below picture depicts the function of SVM in shich the red line is the boundary which SVM tries to find:-
<br>
<div class="imgcap">
<img src="/assets/images/blog5_support-vector-machines/blog5_SVM_margin.png">
</div>
<br>

In machine learning the boundary that separates the examples of the different classes is termed as "Decision boundary", which in context of SVM is known as a 'hyperplane'. In next section we discuss the equation which is mentioned in the image.

The eqationof the hyperplane is given by two parameters, a real-values vector $$ w $$ and input vector $$ x $$ and a real number $$ b $$ like this:
> $$ w*x - b =0 $$

For the new input vector x the equation to predict the class will be given by:
> $$ w*x - b = y $$


We want to find the **maximum margin hyperplane** that divides the group of points x for which $$y = 1$$ from the group of points for which $$y = -1$$. A large margin contributes to the better generalization, that is how the model will work on the new examples in the future. To minimize that we minimize the Euclidean norm of the $$w$$ denoted by $$||w||$$ and given by 

> $$ \ ||w|| = underroot \ \sum W ^2 $$
