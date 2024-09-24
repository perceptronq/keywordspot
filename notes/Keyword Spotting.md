Each KWS model has 3 major components: 
1. feature extraction module
2. DNN
3. posterior handling module
![[keywordspot/notes/Pasted image 20240921212210.png]]

KWS system is always-on.
It should have very low power consumption to maximize battery life.
KWS system should detect the keywords with high accuracy and low latency.

Dimensionality compression of input audio signal is done by converting it to Mel Spectrograms.
![[keywordspot/notes/Pasted image 20240921211633.png]]
In a real-world scenario where keywords need to be identified from a continuous audio stream, a posterior handling module averages the output probabilities of each output class over a period of time, improving the overall confidence of the prediction.

Different techniques for KWS and their cons:
1. Hidden Markov Models(HMMs) and Viterbi decoding: hard to train and computationally expensive during inference
2. RNN: large detection latency
3. DNN: they ignore the local temporal and spectral correlation in the input speech features.
4. CNN: . the drawback of CNNs in modeling time varying signals (e.g. speech) is that they ignore long term temporal dependencies

The primary focus of prior research has been to maximize the accuracy with a small memory footprint model, without explicit constraints of underlying hardware, such as limits on number of operations per inference.

