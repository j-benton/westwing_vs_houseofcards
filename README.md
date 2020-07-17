# Project 3: Web APIs & NLP

## Problem Statement
Can NLP and machine learning accurately determine which subreddit a post came from based solely on the post's text? Does conversation about real-world politics play a role?

## Executive Summary
As the political climate in Washington, D.C. changes, so does that art that seeks to reflect and comment on that climate. Two critially acclaimed political dramas set in the nexus of U.S. government are The West Wing (1999-2007) and House of Cards (2013-2018). Though both dramas began and ended within a 20-year span, the political leadership the series attempt to depict appear radically different. Do these diffences translate into differences in how the shows' respective fans communicate about the shows? Can Natural Language Processing and machine learning parse series' subreddits and determine with a high level of accuracy which comments belong to which fan base? Furthermore, can NLP determine whether these same posts were created before or after Trump's real-world election? 

For get the data for this project, I used Reddit’s Pushshift API to pull text from their website. Pushshift let’s you pull from submissions or comments, so I decided to use both and look at the data all just as text from either subreddit, regardless of post type. I centered my data around the exact time of President Trump's election, pulling the 2000 posts directly before and after that time for both subreddits and for both submissions and comments (total: 8000 documents).

I ran multiple models with this data and landed on Logistic Regression and Multinomial Naive Bayes. For Logistic Regression, I transformed my data using CountVectorizer, which counts how many times each word appears in each document. For Multinomial Naive Bayes, I used TfidfVectorizer, which compares how often a word appearing in a given document appears in the corpus and weights rarer words more than commonly appearing words. In all cases, I used The West Wing's subreddit as my target variable. The accuracy scores for both final models were around 80%.

As you might expect, initial modelling showed that words connected directly to the show (character names, actors, writers, titles, etc) are the strongest correlates to which show subreddit produced the given text. So, I took out every character and actor’s name to see what, if anything, would remain. Unsurprisingly, my accuracy scores dropped substantially, though both were still above the baseline. But some interesting words popped up as having strong correlations to the subreddits, such as "power," "good"/"great"/"best", and explitives. Fans of House of Cards tend to talk more about Trump, though fans of The West Wing tend to use the word "president" more.

I figured if my data had any connection to real world politics, President Trump would surely be at the center of much of that conversation.  However, with the time that I had, my models did not provide insights into how Trump's presence in the political scene affected subreddit conversations about these TV shows. It's possible that there is no effect, save for Trump's name being mentioned in House of Cards. I would like to take more time examining different data subsets from these subreddits (extending the timeline from which I gathered the data), as well as use other current events (i.e. Kevin Spacey's sexual abuse allegations) to see how these subreddits relate to the real world.


## Contents
```
|__ code
|   |__ 1_web_scraping_and_preprocessing.ipynb   
|   |__ 2_model_testing.ipynb   
|   |__ 3_final_modelling.ipynb
|   |__ 4_trump_modelling.ipynb  
|   |__ 5_data_viz.ipynb
|__ data
|   |__ train.csv
|   |__ test.csv
|   |__ submission_lasso.csv
|   |__ submission_ridge.csv
|__ presentation.pdf
```


## Data Dictionary

|Feature|Type|Description|
|---|---|---|
|**text**|*object*|The text data from either subreddit submissions (title and selftext) or comments (body text)|
|**subreddit**|*object*|Source of text data, either House of Cards or The West Wing subreddits|
|**trump**|*int*|Identifies text data post date as being either before (0) or after (1) the 2016 election|
|**submission**|*int*|Identifies text data as being either a comment (0) or submission (1)|
|**created_utc**|*int*|UTC timestamp for submission/comment post|
|**word_count**|*int*|Number of words in document|


## Conclusions
In answer to the problem statement above, NLP and machine learning can predict a text's subreddit source with approximately 80% accuracy using Logistic Regression and Multinomial Naive Bayes. These accuracy scores are largely based on words unique to each show, such as character and actor names. Removing these obvious markers showed more interesting divides between these two shows, such as use of profanity, mention of the current (and controversial) real-world president, and words words like "power" and "war" vs. "great" and "best." However, the scores from these latter models were much less accurate, especially in the case of Logistic Regression:

Original Scores (w/ character names, etc.)
LogReg: 78%
MultiNB: 80%

New Scores (w/o character names, etc.)
LogReg: 61%
MultiNB: 67%

Baseline: 50%

Real-world politics didn't seem to be as predictive as I had hoped, though further analysis is necessary to fully explore this data. For instance, I would like to try using exclusively 3-, 4-, or 5-word phrases and see what phrases are predictive of the subreddits, setting aside optimizing for accuracy (provided that my scores are still above the baseline). I would also like to center my data around different current events relevant to the shows (e.g. go back further to truly test Trump's influence on these subreddit conversations; use fall of 2017 as a data nexus to see if Spacey's sexual abuse case had an impace). Finally, the mechanics of the subreddits themselves should be analyzed: what is the influence of and distinction between submission posts and comment posts in this NLP/machine learning process? How does the relationship between a subreddit's membership count and how active those members are affect these models? There is much more to be explored in this dataset.

## Data Sources

https://api.pushshift.io/reddit/search/submission?subreddit=HouseOfCards
https://api.pushshift.io/reddit/search/submission?subreddit=thewestwing
https://api.pushshift.io/reddit/search/comment?subreddit=HouseOfCards
https://api.pushshift.io/reddit/search/comment?subreddit=thewestwing
