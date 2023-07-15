# mlops-zoomcamp




## 📋 Week 1 Notes: Intro & Environment Setup

When you are designing a machine learning system, your job doesn't end with building the model—and achieving a high accuracy score and a low validation error! For the model to be actually helpful—you will have to consider deploying it, and also ensure that the model's performance does not degrade over time. And MLOps is a set of *best practices* for putting machine learning models into production.

![image](https://user-images.githubusercontent.com/47279635/168582280-52820583-d0bb-4b46-add4-b2fa4c09bc1b.png)

## 🎯 Steps in a Machine Learning Project
The various stages in a machine learning project can be broadly captured in the following three steps:
1. **Design**: In the `design` step, you are considering the problem at hand—to decide whether or not you'll need a machine learning algorithm to achieve the objective.
2. **Train**: Once you decide on using a machine learning algorithm, you `train` the model and optimize its performance on the validation dataset.
3. **Operate**: The `operate` state captures the performance of the model after it's deployed. Some of the questions that you'll answer throughout the course, include:
  - If the performance of the model degrades, can you retrain the model in a cost-effective manner?
  - How do you ensure that the deployed model performs as expected—that is, how do you monitor the model's performance in production?
  - What are the challenges associated with monitoring ML models?


## 📑MLOps Maturity Model

Reference: [MLOps Maturity Model: Microsoft Docs](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)

|Level|Description|Overview|When Should You Use?|
|---|---|---|---|
|0️⃣|No Automation 😢|<ul><li>All code in Jupyter Notebook</li><li>No pipeline, experiment tracking, and metadata</li> </ul>|<ul><li>Academic projects</li><li>Proof of Concept is the end goal, not production-ready models</li></ul>|
|1️⃣|Yes! DevOps😀, No MLOps|<ul><li>Best engineering practices followed</li><li>Automated releases</li><li>Unit \& Integration Tests</li><li>CI/CD pipelines</li><li>No experiment tracking and reproducibility</li><li>Good from engineering standpoint, models are not ML-aware yet!</li></ul>|<ul><li>Moving from proof of concept to production</li><li>When you need some automation</li><ul>|
|2️⃣|Automated Training 🛠|<ul><li>Training pipelines</li><li>Experiment tracking</li><li>Model registry (track of currently deployed models)</li><li>Data scientists work in tandem with the engineering team</li><li>Low friction deployment</li></ul>|<ul><li>When you have increasing number of use cases</li><li>Three or more use cases, you should definitely consider automating!</li><ul>|
|3️⃣|Automated Deployment 💬|<ul><li>Model deployment simplified!</li><li>Prep data >> Train model >> Deploy model</li><li>A/B testing</li><li>Model X: v1, v2 >> v2 is deployed; how to ensure v2 performs better?</li><li>Model monitoring</li></ul>|<ul><li>Multiple use cases</li><li>More mature + important use cases</li><ul>|
|4️⃣|Full MLOps Automation ⚙ |<ul><li>Automated training</li><li>Automated retraining</li><li>Automated deployment</li></ul>|<ul><li>Check if level 2, 3 won't suffice</li><li>Should model retraining and deployment be automated as well?</li><li>Super important to take a pragmatic decision! Do you really need level 4?😄</li><ul>|



#### Useful Resources

1. MLOps Maturity Model from Microsoft: https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model
