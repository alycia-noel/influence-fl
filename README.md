Here, we provide additional information about the experimental setup for the experiments performed in Achieving Flexible Local Differential Privacy in Federated Learning via Influence Functions published at ECML-PKDD'25. We note that we ran all experiments on a Tesla V100-SXM2-32GB.

## Dataset Descriptions
We use three binary datasets to test FLDP-FL: ACS Income, Glioma , and ACS Public Coverage. We note that after distributing the data among the clients according to the process described in Communication-Efficient Learning of Deep Networks
from Decentralized Data, we use a 80/20 split for each client's train $`\mathcal{D}_{i,tr}`$ and test $`\mathcal{D}_{i,te}`$ sets. We assume that the global test set $`\mathcal{D}_{te}`$ is the combination of all clients $`\mathcal{D}_{i,te}`$. 

#### ACS Income:
ACS Income was proposed as an alternate to the traditional Adult tabular dataset and similarly the goal is to predict whether an individual’s income is above $50,000, after filtering the ACS
PUMS data sample to only include individuals above the age of 16, who reported usual working hours of at least 1 hour per week in the past year, and an income of at least $100. The dataset has 10 features (including the label) and 1,664,500 data records. In this work, we select a uniform random sample of 6,000 data points stratified along the labels. We additionally randomize two of the features (RAC1P, SEX) and the label (PINCP) under randomized response. 

#### Glioma:
The Glioma dataset was created by The Cancer Genome Atlas (TCGA) Project funded by the National Cancer Institute. Gliomas are the most common primary tumors of the brain and they can be graded as LGG (Lower-Grade Glioma) or GBM (Glioblastoma Multiforme) depending on the histological/imaging criteria. Clinical and molecular/mutation factors are also very crucial for the tumor grading process. In this dataset, the most frequently mutated 20 genes and 3 clinical features are considered from TCGA-LGG and TCGA-GBM brain glioma projects. The prediction task is to determine whether a patient is LGG or GBM with a given clinical and molecular/mutation features. It has 23 features (including the label) and 839 instances. However, after modifying the Age_at_diagnosis feature to be 10 categories (to allow for randomization under randomized response) and filtering for duplicates, only 416 records remain. We randomize two of the features (Age_at_diagnosis, Race) and the label (Grade) under randomized response

#### ACS Public Coverage:
The goal of ACS Public Coverage is to predict whether an individual is covered by public health insurance, after filtering the ACS PUMS data sample to only include individuals under the age of 65, and those with an income of less than \$30,000. This filtering focuses the prediction problem on low-income individuals who are not eligible for Medicare. The dataset has 19 features (including the label) and 1,138,289 data records. In this work, we select a uniform random sample of 6,000 data points stratified along the labels. We additionally randomize three of the features (AGEP, SEX, RAC1P) and the label (PUBCOV) under randomized response. 

## Model and Training Details
Here we provide more in depth details of the federated training setup. We list the selected hyperparameters for each dataset in the table below which were selected using a grid search over learning rate = {1e-2, 1e-2, 1e-4, 1e-5, 1e-6}, number of federated epochs = {25, 50, 75, 100, 200}, and number of local epochs = {1, 3, 5}. The batch size of 1 was held constant over the grid search as advised by previous research.

| Dataset | Learning Rate | Batch Size | Num. Federated Epochs | Num. Local Epochs |
|---------|---------------|------------| ----------------------|-------------------|
| ACS Income | 1e-4 | 1 |100 | 3|
| Glioma | 1e-3 | 1 | 50| 3|
| ACS Public Coverage | 1e-5|1|50|3|

