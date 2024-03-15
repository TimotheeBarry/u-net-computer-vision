# BE - Semantic Segmentation

## Description
This project contains the implementation of a UNET model for semantic segmentation, and dive into the impact of various architectural choices on the performance of the model.

## Structure
The project is structured as follows:
- `backup/` contains the backup of the model weights of the different experiments
- `data/` contains the dataset (not included in the repository)
- `utils/` contains various utility functions for the project (building the model, loading the data, etc.)
- `notebook.ipynb` is a notebook with the main script of the project, it contains the training and evaluation of the model

## Authors
- Timothée Barry
- Pierre Corbani

### Questions

In this section we investigate the impact of various architectural choices of the proposed U-Net
for semantic segmentation.

***

1. **Max pooling:**
<div class="question">

*Max pooling has been used for down-sizing the input volume. Instead of max pooling, is it possible to simply make use of convolution with a different stride, e.g., 2 ? How does it impact the performance ?*

</div>

<div class="answer">
Yes, it's possible to use convolution with a different stride as an alternative to max pool for downsizing the input volume.
The stride convultion tends to preserve more spatial information than max pooling, which can be beneficial in some cases, but max pooling is more efficient in terms of computation and can reduce overfitting.

We trained a model with a stride of 2 in the convolutional layers instead of max pooling on 100 epochs but the differences in performance are not significative.
Both model reach a similar performance in terms of accuracy and loss.
On a sample of 20 images of the testing set, we can see that the model with max pooling perform better than the model with convolution with a stride of 2 on some images, but the model with convolution with a stride of 2 perform better on other images.
</div>

***

2. **Skip Connections:** 
<div class="question">

*The skip connections have been used in the proposed architecture. Why is this important ?
   Suppose that we remove these skip connections from the proposed architecture, how does it
   impact the result ? Please explain your intuition. Furthermore, the current skip connections
   consist of concatenating feature maps of the encoder with those of the decoder at the same
   level, is it possible to make use of other alternatives, for instance addition, Max or Min operation
   ? Do they impact the results ? Explain.*

</div>
<div class="answer">

Skip connection are important in the proposed architecture to preserve the spatial information. Because of the multiple downsampling and upsampling operations, the spatial information can be lost. The skip connections allow the decoder to access features from multiple scales of the encoder, enhancing feature reuse and potentially improving the model's ability to capture intricate details and semantic information.

Alternatives like addition, max, or min operations are possible but they may not preserve information as effectively as concatenation.


</div>

***

3. **FCN and Auto-encoder:** 

<div class="question">

*The proposed architecture is an auto-encoder like architecture, i.e., composed of an encoder and a decoder. Is it necessary ? Is it possible that we make use of a fully convolutional neural
   network (FCN) consisting of series of convolutions from the input to the output while keeping the
   image size, i.e., width and height, during all the convolutions ?*

</div>

<div class="answer">
FCN architecture consists of a series of convolutions from the input to the output while preserving the image size.

FCN can be used for semantic segmentation (ex: https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1). 

 However, this design might not capture hierarchical features as effectively as U-Net, which exploits the encoder-decoder structure for feature extraction and spatial localization.
</div>

***

4. **Threshold for inference:** 
<div class="question">

*We have taken 0.5 as the threshold to decide whether to classify a pixel as salt or not. However,
   this choice proves not to be the optimal in terms of precision and recall rate for pixels being
   classified salt. Could you explain how we should define the precision and recall rate in our
   business case. Could you propose a strategy to improve both precision and recall ?*

</div>

<div class="answer">

**Precision:** Precision measures the proportion of pixels classified as salt that are actually salt. 
It is calculated as the number of true positive salt pixels divided by the total number of pixels classified as salt.

In our case, 
- TP = pixels correctly classified as salt (1 in the mask and in the prediction)
- FP = pixels incorrectly classified as salt (1 in the prediction and 0 in the mask)

$Precision = TP / (TP + FP)$

**Recall:** Recall measures the proportion of actual salt pixels that are correctly classified. It is calculated as the number of true positive salt pixels divided by the total number of actual salt pixels.

In our case,
- FN = pixels incorrectly classified as non-salt (0 in the prediction and 1 in the mask)

$Recall = TP / (TP + FN)$

To improve both precision and recall, we can adjust the threshold to find the optimal value that maximizes both precision and recall. We can use techniques such as ROC curve analysis to determine the optimal threshold.


<!-- CSS for class anwer:-->
<style>
.answer {
    background-color: rgba(0, 255, 128, 0.1);
    padding: 20px;
    border-left: 6px solid #4CAF50;
}
.question {
    background-color: rgba(0, 128, 255, 0.1);
    padding: 20px;
    border-left: 6px solid #2196F3;
}
</style>


