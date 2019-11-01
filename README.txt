*********************
****  Datasets   ****
*********************
cho.txt and iyer.txt are the two datasets to be used for Project 2. 


*********************
**  Dataset format **
*********************

Each row represents a gene:
1) the first column is gene_id.
2) the second column is the ground truth clusters. You can compare it with your results. "-1" means outliers.
3) the rest columns represent gene's expression values (attributes).

==========================
         README
==========================

This repository includes scripts of K_means, HAC with min, DBSCAN, spectral and GMM
To perform clustering by  specific method on some dataset, follow the instructions below:

1. Copy the datasets to this directory.

2. Run commands in terminal:
   $ python3 K_means.py <file_name> <number of clusters>
   $ python3 hierAgg.py <file_name> <number of clusters>
   $ python3 DBSCAN.py <file_name> <MinPts> <Eps>
   $ python3 spectral.py <file_name> <sigma>

   hard code pi, mu and sigma, convergence threshold, maximum iterations and number of clusters
   $ python3 GMM.py <file_name>



   Replace <file_name> with the file to be processed.
   E.g. python3 K_means.py cho.txt 3 would perform K_means on cho.txt.

3. Observe plots and index