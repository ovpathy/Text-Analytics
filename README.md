# Text Analytics
A project to predict the number of Twitter shares of news articles of a leading news website, based on the content of the articles.

### TF-IDF
Steps taken to represent the articles as Bag of Words:
<li> Normalization of words </li>
<li> Punctuation removal </li>
<li> Stemming </li>
<li> Stop words removal </li>

### Topic Modelling
Topic modelling is done to treat words with multiple meanings differently. Got 7 topics and hence built seven separate models for 7 topics

### Age bias
Articles having similar content will have varying number of shares depending on the age of the articles. Filtered the articles in every topic which have its age greater the mean age for that topic.

### Model
Have used Boosting and ensemble techniques to train the model.
