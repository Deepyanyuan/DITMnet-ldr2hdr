# DITMnet-ldr2hdr
ldr2hdr, fldr2hdr, optical filter, U-net

DITMnet：ldr2hdr via CNNs, detail in paper 《DITMnet: Learning to Reconstruct HDR Image Based on a Single-shot Filtered LDR Image》

original HDR dataset come from online: Fairchild-HDR {http://rit-mcsl.org/fairchild//HDRPS/HDRthumbs.html}, and Funt-HDR {https://www2.cs.sfu.ca/~colour/data/}.

2020-01-09

pipeline:

step 0: generate training pairs by "generate_data.py" or "generate_data_v1.py"

step 1: create network, in this paper, we create a multi-branch and multi-ouput CNNs network, detail in "src/archs.py"

step 2: train this network by "main.py". model type=0
Note that in this paper, we use two datasets, the first had been training and the last dataset (Funt-HDR) had no training.

step 3: test this network by "main.py". model type=1

step 4: predict the results by "main.py". model type=2

step 5: performance comparison by Matlab code


