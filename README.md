This is the code for GAP with transformer coupled with GNN positional encoding, where codes related to loss function and utilization of are referenced from works of Edina's Group.

To run the code, first create a directory named `graphs` (if not exist), and run the following command:
```bash
ipython train.py -- -np 5 -name "mfpt" -feats "mfpt" -agg "mean" -ffl 2 > ./graphs/mfpt.log
```

where `-np` is the number of processes, `-name` is the name of partitions, `-feats` is the type of features, `-agg` is the type of aggregation, and `-ffl` is the number of decoding layers. The log file will be saved in `./graphs/mfpt.log`. 

In the `graphs` directory, there are graphs for clustering and losses for training. The final loss of clustering stores in a file whose name starts with scores in `graphs`.

Besides, when running on a graph with different nodes as 250, add `-n-nodes YourNodeNumber` to the command. For example, if the number of nodes is 547, run the following command:
```bash
ipython train.py -- -np 5 -name "mfpt" -feats "mfpt" -agg "mean" -ffl 2 -n-nodes 547> ./graphs/mfpt.log
```