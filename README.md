# Data Science Project-Set 2024 - *Daud Waqas*

This is a complete set of all the data science projects I underwent in the 2024 calendar year. 

Some of these projects may be part of the *Graduate Certificate of Data Science* course administered by *Harvard Extension School* which I completed at the end of the year, and may be labelled as such. Notebooks are provided as HTML to avoid any potential knitting issues, while reports, reviews and other documentation may be provided as PDFs. I may also include the source Python Notebook (IPYNB) files as well.

## $${\color{blue}Spatio-Temporal Time Series Modelling}$$
**With a Focus on Coral Reef Benthic Group Shifting in the Great Barrier Reef**

I started off this project to tackle a fairly obscure and unexplored topic: *understanding and predicting shifts in **coral reef benthic cover** using data-driven approaches*. This was also my attempt to try and “handle” a much larger and unoptimised dataset. The data, initially provided at yearly intervals, represented the distribution of coral reef covers *(specifically **hard coral cover**, **soft coral cover** and **algae cover**)* across various reef systems throughout the Great Barrier Reef; due to the massive ecological size of the reef, I made an assumption that there would be distinct temporal patterns based on the spatial location that the data was sampled from. This was the basis of a study that was not just focused on fitting **time series**, but also on building a **spatio-temporal model** that would take those location-based dependencies into account.

The data was compiled from the **AIMS** and **eReefs platform**, both of which are managed by the Australian government; **AIMS** was used to source the primary data, and **eReefs** was used to source the auxiliary data. The data was made up of spatio-temporal indexes *(latitude, longitude, date, etc.)* alongside 4 cover values:

* **SOFT CORAL_COVER**
* **HARD CORAL_COVER**
* **ALGAE_COVER**
* **OTHER_COVER**

**COURSE:** This project was done as part of the *Graduate Certificate of Data Science* course administered by *Harvard Extension School*, specifically the *CSCI E-82* course, finished on *Dec 2024 with Certficiate awarded in Jan 2025*.

**MAIN PROJECT DIRECTORY:** *`./"Coral Benthic Shifts with ST-GNNs - PyTorch"`*

**DATA COLLECTION DIRECTORY:** *`./"Coral Benthic Shifts with ST-GNNs - PyTorch"/"Data Collection"`*

## $${\color{blue}Dynamic HyperParameter Optimisation with Optuna}$$

This is a quick project centred around exploring how institutions and researchers may approach hyper parameter optimisation in cases of high model and system complexities, especially since traditional approaches such as **Grid Search** and **Random Search** present some notable limitations. **Grid Search**, while exhaustive, becomes *computationally expensive* as the number of hyperparameters (and their potential combinations) increases. On the other hand, **Random Search** provides a more efficient alternative but lacks systematic exploration, making it unreliable in *scenarios where it is important to understand the “thought process” behind reaching a set criteria of optimised hyperparameters* (think of industries such as quantitative trading, healthcare, etc). To address these challenges, a more dynamic and adaptive solution was required. This project is in the form of a report paired with some visualisations, and explores the effectiveness of **Optuna** and how it addresses the limitations of the traditional methods.

**MAIN PROJECT DIRECTORY:** *`./"Dynamic Hyperparameter Optimisation with Optuna"`*

## $${\color{blue}Research Paper Reviews}$$

**COURSE:** These papers were done as part of the *Graduate Certificate of Data Science* course administered by *Harvard Extension School*, specifically the *CSCI E-82* course, finished on *Dec 2024 with Certficiate awarded in Jan 2025*.

**RESEARCH PAPERS DIRECTORY:** *`./"Research Paper Reviews"`*

### CSPNET: A New Backbone That Can Enhance Learning Capabilities OF CNNs

**Abstract:**

Neural networks have enabled state-of-the-art approaches to achieve incredible results on computer vision tasks such as object detection. However, such success greatly relies on costly computation resources, which hinders people with cheap devices from appreciating the advanced technology. In this paper, we propose Cross Stage Partial Network (CSPNet) to mitigate the problem that previous works require heavy inference computations from the network architecture perspective. We attribute the problem to the duplicate gradient information within network optimisation. The proposed networks respect the variability of the gradients by integrating feature maps from the beginning and the end of a network stage, which, in our experiments, reduces computations by 20% with equivalent or even superior accuracy on the ImageNet dataset, and significantly outperforms state-of-the-art approaches in terms of AP50 on the MS COCO object detection dataset. The CSPNet is easy to implement and general enough to cope with architectures based on ResNet, ResNeXt, and DenseNet.

Source code is at https://github.com/WongKinYiu/CrossStagePartialNetworks

**Link:** https://paperswithcode.com/paper/cspnet-a-new-backbone-that-can-enhance

**Context:** 

I essentially chose this paper due to my curiosity on the potential for highly eﬃcient compute for CNN models, especially in production (ie finalised model) environments. Other than the actual accuracy of the model itself, I believe this is the second factor that companies would deeply consider when it comes to any possible implementation of image classifier solutions.

### Neural Oblivious Decision Ensembles (NODE) For Deep Learning On Tabular Data

**Abstract:**

Nowadays, deep neural networks (DNNs) have become the main instrument for machine learning tasks within a wide range of domains, including vision, NLP, and speech. Meanwhile, in an important case of heterogenous tabular data, the advantage of DNNs over shallow counterparts remains questionable. In particular, there is no suﬃcient evidence that deep learning machinery allows constructing methods that outperform gradient boosting decision trees (GBDT), which are of- ten the top choice for tabular problems. In this paper, we introduce Neural Oblivious Decision Ensembles (NODE), a new deep learning architecture, designed to work with any tabular data. In a nutshell, the proposed NODE architecture generalizes ensembles of oblivious decision trees, but benefits from both end-to-end gradient-based optimization and the power of multi-layer hierarchical representation learning. With an extensive experimental comparison to the leading GBDT packages on a large number of tabular datasets, we demonstrate the advantage of the proposed NODE architecture, which outperforms the competitors on most of the tasks. We open- source the PyTorch implementation of NODE and believe that it will become a universal framework for machine learning on tabular data.

**Link:** https://paperswithcode.com/method/node

**Context:** 

The reason I chose this paper was to see if neural networks did have a place when it comes to dealing with tabular data; in HW3, I actually skipped doing neural networks after FNNs since I determined that it would ultimately be too time-costly versus the gradient boosting models which were already performing quite well. Based on this sentiment, I decided it would be worth my time to see the current state on fitting neural networks onto tabular data, especially since it is often overlooked due to the great performance of machine learning models for the type of data being dealt with.

### Spatial-Temporal Graph Neural Networks (ST-GNNs) for Groundwater Data

**Abstract:**

This paper introduces a novel application of spatial-temporal graph neural networks (ST-GNNs) to predict groundwater levels. Groundwater level prediction is inherently complex, influenced by various hydrological, meteorological, and anthropogenic factors. Traditional prediction models often struggle with the nonlinearity and non-stationary characteristics of groundwater data. Our study leverages the capabilities of ST-GNNs to address these challenges in the Overbetuwe area, Netherlands. We utilise a comprehensive dataset encompassing 395 groundwater level time series and auxiliary data such as precipitation, evaporation, river stages, and pumping well data. The graph-based framework of our ST-GNN model facilitates the integration of spatial interconnectivity and temporal dynamics, capturing the complex interactions within the groundwater system. Our modified Multivariate Time Graph Neural Network model shows significant improvements over traditional methods, particularly in handling missing data and forecasting future groundwater levels with minimal bias. The model’s performance is rigorously evaluated when trained and applied with both synthetic and measured data, demonstrating superior accuracy and robustness in comparison to traditional numerical models in long-term forecasting. The study’s findings highlight the potential of ST-GNNs in environmental modelling, oﬀering a significant step forward in predictive modelling of groundwater levels.

**Link:** https://www.nature.com/articles/s41598-024-75385-2

**Context:** 

Since this paper has to do with one of my key interests in data science (spatio-temporal modelling), I decided on exploring a type of graph-based NN that would be optimised for spatio- temporal datasets, especially those that may have temporally varying feature-sets. Though I have not yet finalised the exact set of methods which I am to cover in the final project, I have decided that I am at least going to cover STARIMA and ST-GNNs. In terms of complexity, ST-GNNs have a much more complex architecture behind them, which is why I thought they would be most relevant to cover for this paper extension. I did originally decide to do the oﬃcial ST-GNN paper (refer to the first link in the alternative papers), but it was too mathematical so I instead pivoted to a case study focused on a hydrological use case (groundwater monitoring); this case study is relatively similar to what will be covered in my final project, so it’ll be good to learn ST-GNNs from an “environmental modelling” perspective.

**Alternative papers:**

*(Since there are many papers on spatio-temporal time series modelling, there were some other papers which I had in mind as well):*
https://arxiv.org/abs/2110.02880 (the original; too mathematical for this extension)
https://arxiv.org/abs/2001.02250
https://arxiv.org/abs/2312.12396
https://arxiv.org/abs/2205.13504

*(This YouTube video also helped in getting some of the fundamentals figured out):*
https://www.youtube.com/watch?v=RRMU8kJH60Q