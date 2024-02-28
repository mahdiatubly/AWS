## Introduction to MLOps

* To implement quality control measures at the model registration step. If a model meets baseline performance metrics, it can be registered with a model registry. A model registry can be used as a quality gate. A model registry is a mechanism that is used to:
  * Catalog models for production.
  * Manage model versions.
  * Associate metadata, such as training metrics, with a model.
  * Manage the approval status of a model.
 

## Initial MLOps

* Deep Learning Containers provide optimized environments with TensorFlow and MXNet, Nvidia CUDA (for GPU instances), and Intel MKL (for CPU instances) libraries.
* SageMaker Studio environments do not support the elevated permissions needed to access the Docker daemon for building a Docker image. A separate build environment is needed to extend or build containers. The Amazon SageMaker Studio Image Build Command Line Interface (CLI) lets you build Amazon SageMaker-compatible Docker images directly from your Amazon SageMaker Studio environments.
* The build CLI automatically sets up a reusable build environment that you interact with by using high-level commands. The CLI orchestrates the build workflow using AWS CodeBuild and returns a link to your Amazon Elastic Container Registry (Amazon ECR) image location.
* Accessing the Docker daemon within SageMaker Studio:
  1. Assume execution role
  2. Package directory and upload to S3 bucket
  3. Build the Docker image using AWS CodeBuild
  4. Push the image to the ECR repository and return URI

