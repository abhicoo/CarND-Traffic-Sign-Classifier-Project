#**German Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: data_distribution.png "Data Distribution"
[image2]: data_visualization.png "Data Visualization"
[image3]: processed_data_visualization.png "Processed Data Visualization"
[image4]: graph_run.png "Model Architecture"
[image5]: cost.png "Cost"
[image6]: ./my_images/1.jpg "Traffic Sign 1"
[image7]: ./my_images/2.jpg "Traffic Sign 2"
[image8]: ./my_images/3.jpg "Traffic Sign 3"
[image9]: ./my_images/4.jpg "Traffic Sign 4"
[image10]: ./my_images/5.jpg "Traffic Sign 5"
[image11]: ./my_images/6.jpg "Traffic Sign 6"
[image13]: ./my_images/8.jpg "Traffic Sign 8"
[image14]: probability_distribution_end_of_speed_limit_60.png "probability_distribution_end_of_speed_limit_60"
[image15]: probability_distribution_elderly_crossing.png "probability_distribution_elderly_crossing"
[image16]: feature_map.png "Traffic Sign 8"

Here is a link to my [project code](https://github.com/abhicoo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

![alt text][image1]

Above is an exploratory visualization of the train data set. It is a bar chart showing how many data are there for each class in training set. As we can see the number of
data for each class is not same and there is some huge difference. Some classes have much more examples than other. We can use data augmentation techniques to fix this.

![alt text][image2]

Above is image of the images present in training data set. As we can see images are in different contrast and brightness. We can apply image processing techniques so that
the CNN can learn better representation(features) from data.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

We will convert image to grayscale since color has very little thing to do with classifying the traffic signs. The symbols structure in traffic signs is important feature.
We will also normalize the image so that we are not passing very high values of classifier. It helps model in learning parameters easily.
We will also apply histogram equalization technique so that we have better contrast in our images.

![alt text][image3]
Above is image showing how data looks afters processing.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Batch Normalization					|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	      | 1x1 stride, valid padding, outputs 10x10x16    |
| Batch Normalization					|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten					|												|
| Fully connected		| 400x120        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| 120x84        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| 84x43       									|
| Softmax				|      									|
| Cross Entropy				|      									|
| Optimizer			|      									|

![alt text][image4]
Above is image showing final model generated from tensorboard
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model i used Adam Optimizer. 
The optimizer needs to optimize something.
The thing which we are going to optimize is a loss function called Cross Entropy.
Cross Entropy Mesaures how far our predicted probability distribution is from actual probability distribution.

The batch_size is set to 128 which is how much data we will process in each iteration to compute gradient and adjust or tune weights.
The epochs is set to 15 which was calculated through trial and repeat method at point after which accuracy stops to improve.
The learning rate is set to 0.001 which was calculated through trial and repeat. We accept the learning_rate which redues the loss most.

![alt text][image5]
Above is image showing how cost changed over every epoch generated from tensorboard

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 98%
* test set accuracy of 96.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
	I started with the same architecture as that of LeNet-Lab class. I choose this as starting point because it gave more that 95% accuracy on the dataset it was trained on. 
* What were some problems with the initial architecture?
	When i used with the same model with Traffic Sign data set the accuracy was less that 90%. The model was not using some of the latest techinques which helps in better
	learning(Batch Normalization) and better generalization(Dropout);
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	The thing which helped to improve the most most was doing better preprocessing. The historgram equalization really helped. 
	I added batch normalization layer before the activation layers so that we have better learning as we are proving normalized data to each layer not only the input.
	I also added dropout layers to help model not overfit on training data and perform better on validation and test data.

* Which parameters were tuned? How were they adjusted and why?
	Epochs, Batch Size, Learning rate, Keep Probability(Dropout) were tuned.
	The mu and sigma was also adjusted for better weights initialization. The sigma was set to 0.01. The way i reached to this value is by computing initial cost and checking
	that its value is near to -ln(1/n_classes). Initally the model should give equal probability for each class. This also helps us tell whether the model architecture was
	implemented correctly.
	The batch_size is set to 128 which is how much data we will process in each iteration to compute gradient and adjust or tune weights.
	The epochs is set to 15 which was calculated through trial and repeat method at point after which accuracy stops to improve.
	The learning rate is set to 0.001 which was calculated through trial and repeat. We accept the learning_rate which redues the loss most.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
	Adding Batch Normalization and Dropout Layer helped in better predictions.

If a well known architecture was chosen:
* What architecture was chosen?
	I used basic version of LeNet model with some adjustments like Batch Normalization and Dropout.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	The model is having for than 95% accuracy on all the datasets. Since the accuracy difference is not much and high the model has learnt and generalized well.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11] ![alt text][image13] 

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery Road			| Slippery Road      							|
| Speed limit(60km/h)    			| Speed limit(60km/h)										|
| Road Work					| Road Work						|
| Speed limit(70km/h)      		| Speed limit(70km/h)					 				|
| Traffic Signals		| Traffic Signals     							|
| End of Speed Limit (60km/h)		| End of Speed Limit (80km/h)    							|
| Elderly Crossing		| Speed limit(70km/h)    							|


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 67%. 
The network actually gave 100% accuracy on the traffic signs which are present in our training dataset.
The End of Speed Limit (60km/h) is not present in our dataset at all but the network came very near to a class present in training data set End of Speed Limit (80km/h).
The Elderly Crossing was misclassifed totally.
Lets check the probabilty of top 5 classes for misclassifed images.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image11]
![alt text][image14]


The End of speed limit(60km/h) is classified as End of speed limit(80km/h). Our training data didnot has End of speed limit(60km/h) but the network was able to recognize it as end of speed limit sign because of the stripe over the number which it has seen in training was said to be a end of speed limit sign. The network also gave some probability to speed limit 30km/h as there is a number behind the stripe but because of the stripe the number feature detection was not correct. The network also gave some probability to end of all speed and passing limit because the end of all speed and passing limit is just a stripe without any number with it.

![alt text][image13]
![alt text][image15]

The elderly crossing sign is not classified properly at all. The network gave some probability to road narrows on right class as the structure in image has right curve which is narrowing down.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image16]
The first conv layer was able to detect edges in the image.

