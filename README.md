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

### Off-time (12-18 OCT)

### Next Meet (19-25 Oct anytime):
>__To us: "feel free to add more tasks as needed!"__
- Start transforming movie reviews into vectors (__*TF-IDF*__)
- Work on getting the genres for each of the movies using genre-ids.
- Implement multi-class classification (**_use softmax (not sigmoid)_**)

> May need to work more days during this period (**Project Milestone 1 due!**).

