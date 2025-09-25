Description:
This repository implements a multi-stage ensemble learning approach for the "Fake or Real: The Impostor Hunt" challenge. It combines multiple base models, such as Support Vector Machines (SVM), Neural Networks, and Torch models, with stacking meta-learners to enhance performance. The process consists of three main stages:

Stage 1: Trains multiple base models (SVMs, Averaged SVM, Torch NN).

Stage 2: Trains meta-learners (Logistic Regression, Random Forest, HistGradientBoosting, and Torch NN) on residuals from Stage 1.

Stage 3: Trains higher-level meta-learners on Stage 2 residuals.

Additionally, the repository includes support for optional GPU acceleration using cuML and techniques for generating extra meta-features from Level-1 out-of-fold predictions. The ensemble approach also supports weighted blending of the best Stage-2 meta-learners in Stage-3.

The model leverages the SciBERT pre-trained language model for generating embeddings and improving classification accuracy. The final output is a robust classification model for determining whether the provided text pairs are real or fake, based on the features extracted from the BERT embeddings.

Features:

Support for GPU acceleration with cuML (if available).

Multi-stage stacking ensemble with performance boosting meta-learners.

Optional weighted blending of top meta-learners in Stage 3.

Use of SciBERT model for text embeddings.

Cross-validation and model evaluation using accuracy and R^2.

Usage:

Train the model on the provided dataset, or use the pre-trained model and embeddings for inference.

Customize the base models, meta-learners, and embedding parameters as needed.
