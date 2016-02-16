---
layout: post
title:  "Verbose Explanation of Implementating Structured Perceptron"
date:   2015-09-15 23:55:00
categories: jekyll update
---

In ACL-IJCNLP 2015, there was [a nice tutorial about "Scalable Large-Margin Structured Learning: Theory and Algorithms"][tutorial], which mainly focused on structured perceptron.

In the tutorial, [Haul Daume's PhD thesis][PhD thesis] was referred.

But I came up with a question; "Is this REALLY calculating the averaged perceptron? Can I prove it or explain it verbosely?".
The explanation on the slide was not verbose enough for me to fully understand it.

Let's start from referring to the average structured perceptron from Haul Daume's PhD thesis:

![My helpful screenshot]({{ site.baseurl }}/assets/averagedStructuredPerceptron.png)


What is this 
$$ w_0 - w_a / c $$

[tutorial]:      http://acl2015.org/tutorials-t6.html
[PhD thesis]:    http://www.umiacs.umd.edu/~hal/docs/daume06thesis.pdf
