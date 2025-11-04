# CS486-movie-sentiment-analysis
*An agent to be designed to help out with movie selection and ease of choice-making with everything related to movies.*

### Team:
- **Cora Schmidt**
- **Ankit Mukhopadhyay**


## Plan
### Basic Idea: 
Movie Recommendation Agent. (Experiment with classification of the movies into emotional classification categories)

### Dataset: 
[TMDB for movies](https://www.themoviedb.org/?language=en-US)
(We will use more datasets as needed, potentially for reviews of the movies)

### NLP Methods:
- Text preprocessing (tokenization), 
- TF-IDF for classification (or Naive Bayes/Logistic Regression)
- Map like data structure to map words to each review/description/genre of a movie.
- Regex (if needed)
- Stop Word Removal
- Bag of Words
- K-means clustering for classification (potential option- maybe, maybe not)?

### AI Agent Implementation:
Model API will be used to call tools specifically, which we will implement. It will not be used to generate any responses on its own.

### Notes:
*We’ll make an API call from the TMDB site. We will get the response as a JSON format, with the description and the title of the movies, and then we will extract the specific information we need, like the description, genre, title, rating etc.*

*Take data, classify in multiple ways, classification task can determine what kind of movie it is, overview, popularity.*
*According to the user's interest, have top n movies based on popularity or vote.*
I*F there is a space complexity problem: remove movies lower than the threshold of popularity.*

*Given a bunch of movies, get recommendations for closest to similar movies based on that (potential feature)*

## **Meeting Plan Schedule**
### ~~Next Meet (5-11 Oct anytime):~~
> **Completed Oct 10**
- ~~GitHub repo.~~
- ~~Basic json extraction into dataframe using pandas (testing).~~
- ~~Text Preprocessing (__*lowercasing, tokenization, stop word removal, punctuation removal, stemming*__)~~
- **TO-DO**:
    - [x] **Meet and complete the requirements.**

### Off-time (12-18 OCT)
> Happy Birthday Cora *(15th Oct)*

### Next Meet (19-25 Oct anytime):
> **Completed Oct 21**
- ~~Start transforming movie reviews into vectors (__*TF-IDF or Word2Vec*__)~~
- ~~Start implementing genre-ids to genre pipeline (*fetch genres from site, for the corresponding genre-ids*)~~

> 26th Oct (**Project Milestone 1 due!**).
- **TO-DO**:
    - [x] **Meet and complete the requirements.**

### Next Meet (26 Oct - 1 Nov anytime): 
> **Completed Oct 27**
- ~~Apply text classification, start with **label encoding** based on genre from the dataframe.~~
    - ~~*More label encoding based on different criterias to be applied later as needed.*~~
- ~~Implement the model, train and test it for classification based on accuracy.~~
- **TO-DO**:
    - [x] **Meet and complete the requirements.**

### Next Meet (2-8 Nov anytime):
> **Completed Nov 3**
- ~~Handle multiple pages for json output into the dataframe (__*needed for potential better accuracy, with more data*__).~~
- ~~Maybe implement the first genre for each of the movies, do **not** implement all the genres as separate classes (__*potential implementation to improve the accuracy*__), maybe add the implementation of multiple genres for a movies, classify it under different genre in that case.~~
- **TO-DO**:
    - [x] **Meet and complete the requirements.**

### Next Meet (9-15 Nov anytime):
- Work on getting the genres for each of the movies. Eventual shift from *genre_ids* to *genres*.
- Experimenting on the representation of each of the overviews, maybe adding instead of finding the mean (**_for potential better accuracy_**).
- Implement multi-class classification (**_use negative sampling or softmax (not sigmoid)_**) (__*potential better accuracy implementation*__).
- **TO-DO**:
    - [x] **Meet and complete the requirements.**

## Problems:
- Classification of multiple genres of a movie as a separate class, which leads to the creation of a huge number of classes, reducing the accuracy.


