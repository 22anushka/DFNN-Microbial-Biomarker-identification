# DFNN-Microbial-Biomarker-identification

This repository contains code files and relevant datasets for [Microbial Biomarkers Identification for Human Gut Disease Prediction using Microbial Interaction Network Embedded Deep Learning](https://thesai.org/Publications/ViewPaper?Volume=14&Issue=6&Code=IJACSA&SerialNo=135).

Sivakumar, A., Syama K and Arul, A. (2023). _“Microbial Biomarkers Identification for Human Gut Disease Prediction using Microbial Interaction Network Embedded Deep Learning”._ IJACSA, 14(6).
Presented at International Conference on Graphs, Networks and Combinatorics (ICGNC 2023)

## TLDR 
[View the PPT](DFNN.pdf)

Inflammatory Bowel Disease (IBD) and Colorectal Cancer (CRC) are global diseases, affecting millions of humans around the world. IBD, as of 2020, plagues 6 million globally and is on a steady rise and Colorectal Cancer is the third most frequently maligned cancer in the world.
The purpose of this work is to identify prominent and meaningful biomarkers from metagenomic datasets of diseases such as IBD and CRC. The approach is to construct an informative Microbial Interaction Network using the tool MAGMA which will help capture the underlying biological associations between the microbial entities, and then embed the resulting network into a Deep Feed-Forward Neural Network model for the purpose of identifying microbial biomarkers.


Methodology
The workflow of the methodology is as follows:

1) The metagenomics dataset is obtained from relevant studies
2) The dataset is reduced by applying a prevalence measure threshhold and a sampling read threshold (note to self: used to filter out microbes present in less than threshhold number of samples)
3) Then, a Microbial Interaction Network is constructed using the tool MAGMA on the reduced dataset. Further, the reduced dataset is randomly split into 80:20 training and testing datasets.
4) In the next step, the MIN constructed is embedded into the feed-forward neural network architecture and is trained on the training dataset. In this step, feature selection and feature importance scoring takes place during training.
5) The top subset of features which rank highly on their feature importance score form the set of informative biomarkers. For model validation, these top features are then put through various classifiers namely Support Vector Machine, Deep Forest (DF), Random Forest (RF), Multi-Layer Perceptron (MLP), and XGBoost (XGB). Using the test dataset, they are evaluated on Area Under Curve score, Accuracy, and F1- measure which helps validate how informative and meaningful the biomarkers selected are.
Finally, the best performing set of biomarkers selected by the proposed methodology are cross-validated against biological studies on the same dataset.

A deeper dive into the methodology
The reasons for following method of constructing the MIN and then embedding it is as follows:
1) Most microorganisms do not live in isolation and thrive in communities while forming interactions and establishing ecological relationships  that shape microbial abundances. By constructing a MIN, the interplay between the environment and microbial populations can predictively modeled as a network.
2) To construct these networks, we make use of absolute abundances rather than relative abundance. This way, we are able to avoid compositionality bias (Note: The change in proportion of one variable may bring about an increase or decrease in another variable regardless of if the two variables are correlated since the total abundance must sum up to 1.) thereby avoiding spurious or confounding correlations.
3) By embedding this network into the model, we generate a sparse first layer connection instead of a fully connected first layer. We are able to represent the high dimensional vector representation of variables in low dimensions while preserving relevant information like the topology of the network and the relationship between nodes. This way, we can reduce overfitting, and noise help improve the reliability of the network. 


Microbial Interaction Network Construction tool: MAGMA
A. Cougal, et al. (2019) proposed a method for the construction of MINs called MAGMA, short for Microbial Association Graphical Model Analysis.
MAGMA makes use of a Gaussian copula mathematical graph model. It is able to account for data flaws such as noisy structure, overdispersion, and zero-count values, and can also handle compositionality bias. The main feature of MAGMA is that it integrates covariates (characteristics of the participating variable) which improves the quality of inference of the categorical variables.


Network architecture:
The neural network model is a feed forward deep model consisting of an input layer, 4 hidden layers and an output layer, as seen in the figure. The input layer has number of neurons = number of features and it connects to a sparse layer having the same number of neurons. The edges are connected based on the adjacency matrix formed by the microbial interaction network. (Note: The first hidden layer has 128 neurons, second has 32, and third has 8 before connecting to the output. Each of these hidden layers except the graph embedding layer have a dropout of 0.5, and the model has a learning rate of 0.0001 and makes use of Adam Optimizer.) Upon multiple rounds of trials, this architecture was successful in resulting in high auc score, high accuracy, and minimum loss.

Feature Importance Scoring
The feature importance score is given on the basis of the graphical connect weight method. The relative importance of each feature is scored on the basis of the sum of absolute values of the weights directly related to that feature or neuron. 

Findings and Results:
Various other popular Microbial Interaction network construction tools such as SparCC, and Spiec-Easi were used and embedded into the deep model for comparison with the proposed methodology. The top features were put through the following classifiers for performance review for both the IBD and CRC dataset. The baseline condition of no feature selection was also applied for the two datasets.

As seen in the IBD dataset, the maximum average value of **auc 0.863, accuracy 0.839, and f1 measure 0.897** was achieved with the combination of **MAGMA with DF classifier for top 300 features**.
Similarly, as seen in the CRC dataset, the maximum average value of** auc 0.837, accuracy 0.768, and f1-measure 0.757** was achieved with the combination of **MAGMA with DF classifier for top 400 features.**

When compared with an existing MIN and graph embedding methodology as proposed by Zhu Q. et al, it is seen that for both the IBD and CRC datasets, the maximum average values achieved are less than the maximum average values achieved by our proposed methodology.

Additionally, upon cross validating the top selected biomarkers against existing biological studies on the same dataset, we can observe similarities in the selected biomarkers and those validated in the studies to be contributing to the presence of disease, with the common biomarkers highlighted in bold. 

In conclusion, MAGMA MIN emphasizes the underlying biological process especially through inclusion of convariates, consideration of multivariate associations, and partial associations. Among other tools, MAGMA also showed the most tempered output. Embedding the MIN helped deal with high levels of noise and overdispersion making for a more reliable feature selection.
Finally, the proposed methodology achieved the highest average evaluation scores when classified using DF across both the datasets.

