# StratFormer

StratFormer is a report studying the impact and relevance of language models architecture on an original task : NFL data. 

This repo contains the draft version of the report and will have proper code added soon. 

In the meanwhile, if you are interested by the project do not hesitate to reach out to me !

Report : https://github.com/samchaineau.github.com/StratFormer/StratFormer.pdf

Contact : sam.chaineau@gmail.com


## Abstract 

This report presents "StratFormer”, an attention-based model applied on NFL data. The model extracts
information from trajectories performed by players on the field during a play. To train the model, we ask it to
guess the team and position of the player. The model also has to complete the trajectory, and decides whether
two trajectories were drawn from the same play.

The model is trained on the NFL Big Data Bowl data set of 2021. It yields an 80% accuracy for the
team classification task, 40% for the position classification task (with the correct position being in the top 3
predictions 67% of time). The correct path is completed correctly 52% of time (with the correct completion
being in the top 3 predictions 84% of time). On the task of detecting if two trajectories were made during
the same play, the model achieves an accuracy of 54%.

We later assess those embeddings’ quality on a concrete application with the objective of classifying
passing plays' results. While using only the trajectories and no other information, we achieve a 57% accuracy
on a three-classes classification problem. Benchmarks found on Kaggle are around 40 to 55% but made on
a different data set. This performance should be back-tested.

We highlight some limitations of the findings as the work has been done on a relatively small, sparse
and unbalanced data set. Further work may affect first findings. However we clearly prove the relevancy of
attention-based models for sports analytics, especially in the NFL.


![Attention examples from trajectories, first attention layer](https://github.com/samchaineau/StratFormer/blob/main/resources/images/Example_1_Attention_1.png?raw=true)
![Attention examples from trajectories, second attention layer](https://github.com/samchaineau/StratFormer/blob/main/resources/images/Example_1_Attention_2.png?raw=true)
![Attention examples from trajectories, third attention layer](https://github.com/samchaineau/StratFormer/blob/main/resources/images/Example_1_Attention_3.png?raw=true)
