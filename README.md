# Purpose
This is the code for detecting ASD using LSTM and CGCN based on fMRI data (my MSc_Project), which I use to find an internship. The project mainly using ROI time series as node feature and extracting temporal feature from LSTM to rebuild the graph as CGCN model input to use spatial and temporal feature from fMRI to do the detection, the best performance is 88.2% under cc200 atlas with default hyper parameters set in the project in 7:3 training, validating and testing, and 5-fold cross validion in training and validating set. The CGCN reffered [BrainGNN](https://github.com/xxlya/BrainGNN_Pytorch/)by Li et al.[1], [Tutorial](https://medium.com/stanford-cs224w/gnns-in-neuroscience-graph-convolutional-networks-for-fmri-analysis-8a2e933bd802)[2] and [CGCN](https://direct.mit.edu/netn/article/5/1/83/97525/Graph-convolutional-network-for-fMRI-analysis) by Wang et al.[3].
# Overview
Below image is the overview of this project
![Overview](https://github.com/user-attachments/assets/efd94c25-b87a-42d5-bf98-3527bb1dc3b3)
# Data Source
All data are from [ABIDE I](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)
# How To Run
* First step: respectively download the pre-processed ASD and TDC ROI time series data follow [preprocessed-connectomes-project
/
abide](https://github.com/preprocessed-connectomes-project/abide), the number of subjects is 884.
  * parameters used to download: filt_globals, cc200 and cc400 atlases
* Second step: run the "calculateCorrelationMatrix.py" to generate FC matrices, please reset the dataset path as yours if it can't work.
* Third step: directly run the "KfoldLSTMEdgeGCN.py" if it won't raise error, else run this after running "timeSeriesPartialCorrelation.py".
# References
[1]Li, X. et al. (2021) ‘BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis’, Medical Image Analysis, 74, p. 102233. doi:10.1016/j.media.2021.102233.<br>
[2] Hough, S. (2022) GNNS in neuroscience: Graph convolutional networks for fmri analysis, Medium. Available at: https://medium.com/stanford-cs224w/gnns-in-neuroscience-graph-convolutional-networks-for-fmri-analysis-8a2e933bd802<br>
[3] Wang, L., Li, K. and Hu, X.P. (2021) ‘Graph Convolutional Network for fmri analysis based on Connectivity Neighborhood’, Network Neuroscience, 5(1), pp. 83–95. doi:10.1162/netn_a_00171.
