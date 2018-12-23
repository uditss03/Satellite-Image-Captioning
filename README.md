# Satellite-Image-Captioning 
Recent improvements in satellite imaging technology
have given rise to new opportunities for more accurate
quantification of both broad and minute changes on Earth,
including deforestation. <br />


<img align="left"  src="https://user-images.githubusercontent.com/34382779/50384897-bf6d3100-06f1-11e9-98b1-5c88dd84b44d.jpg">

Almost a fifth of the Amazon rainforest has been cut down in the last 40 years.To estimate and analyse forest this application is developed. Satellite images are trained on deep convolutional neural networks (CNNs) to learn image features and used multiple classification frameworks including gate recurrent unit label captioning and sparse_cross_entropy to predict multiclass, multi-label images. By fine tuning an architecture consisting of encoder of pre-trained VGG-19 parameters trained on ImageNet data together with the GRU decoder.
 
![filled](https://user-images.githubusercontent.com/34382779/50384941-7e295100-06f2-11e9-9d2d-36321ffc2655.jpg)
<br />
To run Satellite-Image-Captioning:<br />
1. Download Anaconda python<br />
2. Install Tensorflow<br />
3. Install Keras<br />
4. Download this repo.
5. Download encoder weights form https://drive.google.com/open?id=1ArRWH6lIXqiPdlt9vpveqeboiLOVbKaF and decoder weights from https://drive.google.com/file/d/1wNlh8Ds6p3Hu4SnJoqSkL1_qaVB3UaVb/view and save in same folder.<br />
6. Run run_this.py


