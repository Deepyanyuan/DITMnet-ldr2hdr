# DITMnet-ldr2hdr
ldr2hdr, fldr2hdr, optical filter, U-net

DITMnet：ldr2hdr via CNNs, detail in paper 《Method for Reconstructing a High Dynamic Range Image Based on a Single-shot Filtered Low Dynamic Range Image》accepted in Optics Express

original HDR dataset come from online: Fairchild-HDR {http://rit-mcsl.org/fairchild//HDRPS/HDRthumbs.html}, and Funt-HDR {https://www2.cs.sfu.ca/~colour/data/}.

2020-01-09

We have made further adjustments and optimizations based on the previous manuscripts and code. The details are as follows:


1. Added virtual filter camera code for simulating physical optical filters, see "generate_data.py" or "generate_data_v1.py"


2. Further optimized the code of the network structure to make it more clear and concise, see "src/archs.py"


3. Integrated training, testing, and prediction code, see "main.py"


pipeline:

step 0: generate training pairs by "generate_data.py" or "generate_data_v1.py"

step 1: create network, in this paper, we create a multi-branch and multi-ouput CNNs network, detail in "src/archs.py"

step 2: train this network by "main.py". model type=0
Note that in this paper, we use two datasets, the first had been training and the last dataset (Funt-HDR) had no training.

step 3: test this network by "main.py". model type=1

step 4: predict the results by "main.py". model type=2

step 5: performance comparison by Matlab code


