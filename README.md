# 02456-Deep-Learning-project

This project is about applying unsupervised deep learning for food industry automation, see UnsupervisedDeepEmbeddingForClusteringAndAnomalyDetection.pdf.

### Abstract
Currently in food industry most quality assurance tasks are performed manually by workers.  Since most of these tasks involve  visual  assessment  they  are  good  candidates  to  be automated  through  machine  learning.   This  project  investigates an unsupervised deep learning approach for clustering and anomaly detection on images of pork chops.  Clustering based on Unsupervised Deep Embedding for Clustering Analysis was applied, and modifications to the base algorithm for anomaly detection were implemented.   Clustering approach did  not  achieve  significant  performance,  2-class  anomaly detector achieved 10% accuracy increase over the naive baseline.   Results suggest that for this problem an unsupervised approach is unlikely to achieve comparable performance withsupervised methods.



### Logs

Note: we are missing a jupyter notebook for reproduction since the AWS instances we used were commandline-only. Instead refer to DEC_aws_kotlet.py file and logs.

two_class_anomaly_detector.log is the output for the 2-class detector achieving 76+% accuracy when best possible cutoff value is chosen, 74+% when selecting cutoff via some unsupervised means.

mnist_orig_log is the output from reproducing 84% unsupervised clustering accuracy on MNIST referenced in the original DEC paper.

Original DEC code from https://github.com/fferroni/DEC-Keras

