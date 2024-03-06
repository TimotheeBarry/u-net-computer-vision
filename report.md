### Questions
In this section we investigate the impact of various architectural choices of the proposed U-Net
for semantic segmentation.
1. **Max pooling:** Max pooling has been used for down-sizing the input volume. Instead of max pooling, is it
possible to simply make use of convolution with a different stride, e.g., 2 ? How does it impact
the performance ?

2. **Skip Connections:** The skip connections have been used in the proposed architecture. Why is this important ?
Suppose that we remove these skip connections from the proposed architecture, how does it
impact the result ? Please explain your intuition. Furthermore, the current skip connections
consist of concatenating feature maps of the encoder with those of the decoder at the same
level, is it possible to make use of other alternatives, for instance addition, Max or Min operation
? Do they impact the results ? Explain.

3. **FCN and Auto-encoder:** The proposed architecture is an auto-encoder like architecture, i.e., composed of an encoder
and a decoder. Is it necessary ? Is it possible that we make use of a fully convolutional neural
network (FCN) consisting of series of convolutions from the input to the output while keeping the
image size, i.e., width and height, during all the convolutions ?

4. **Threshold for inference:** We have taken 0.5 as the threshold to decide whether to classify a pixel as salt or not. However,
this choice proves not to be the optimal in terms of precision and recall rate for pixels being
classified salt. Could you explain how we should define the precision and recall rate in our
business case. Could you propose a strategy to improve both precision and recall ?