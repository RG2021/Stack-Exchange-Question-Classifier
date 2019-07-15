# Stack Exchange Question Classifier
This project is made using the concepts of Machine Learning, Data Analysis and NLP with the result accuracy of 85.36%. In this project I had taken the sample Dataset of Stack exchange having around 20K data samples for the training purpose. Each data contains 3 entries Question, Excerpt and the Topic which we have to identify by using the Question and the Excerpt.

# Tools Used:
1. Python
2. Numpy
3. ScikitLearn Library.

# Step 1.
Import the all required libraries as mentioned in the code.

# Step 2.
Read the training file in json format containing question, excerpt and topic columns. In this code I had only considered the excerpt part for the training of the model but you can also consider the question and experpt part together just by joining the each question with the excerpt part.

# Step 3.
Read the test file containing question and excerpt columns in the json format and then store the each excerpt part in the test array and then read the test_output file containing the topics for our test data that will be used for the validation of our model.

# Step 4.
Now I had define the TdifVectorizer that is used to convert our text data into the feature matrix. Note I had defined some parameters in the TdifVectorizer those are used to improve our feature matrix like 'analyzer'='word' will only consider the words and not the numbers or special characters as that would be of no use in our prediction and 'stop_words'='english' will remove all the words that that will have no meaning or contribution to our prediction model like 'is', 'the', 'we', 'has' etc. To know more about it as does it convert the text into feature matrix and what other parameters are used for refer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

After defining I had just fit and transform my train array into the feature matrix and then also transform the test array.

# Step 5.
Then in next step I had converted our topic array to the numbers using the LabelEncoder function from the scikit-learn library as our ML model can only work on the numbers.

# Step 6.
Now I had define the ML algorithm(MultinomialNB) and fit it on the training data and then predict the values using the test data. Then I had use the inverse_transform function that will convert back the predicted numbers into the topics.

# Step 7.
And finally then using the accuracy score function i had calculated the accuracy of our model by comparing the test_output values to the predicted values.

So after using this I had got an accuracy of 85.36%.

# Improvements
You can improve your model accuracy by making some changes in this code like
 1. You can tune the parameters of the TdifVectorizer to get a better and more efficient feature matrix 
 2. Or you can use the different ML algorithm like Support Vector Machine(SVM) that is a more powerful algo in terms of Text Classification.

