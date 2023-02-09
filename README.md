# Deep-Matrix-Factorization-Recommendation-Systems
## Introduction
While the initial motivation may seem to be derived from the popular “Netflix prize problem”, recommendation systems find their use and application in many other fields such as “Business-to-customer” in E-commerce, sports game results prediction, gene expression prediction, online learning recommender systems, etc.

The types of recommendation systems approaches include

-   Collaborative filtering: Collaborative filtering predicts
    suggestions for a particular by observing and processing the
    preferences of many users (collaborating). It works on the
    assumption that if person A has the same preference as person B on a
    product, A is more likely to have B’s opinion on a different product
    as well

-   Content-based filtering: It recommends other items similar to what
    the user has previously liked/preferred. Future predictions are
    based on their previous actions or explicit feedback

-   Hybrid approaches: This approach combines collaborative filtering,
    content-based filtering, and other approaches

Collaborative filtering is the basis of the Recommender System used in
this project, and it predicts suggestions for a particular user by
observing and processing the preferences of many users (collaborating).
It can be achieved by applying Matrix Factorization, where a sparse
matrix contains the user ratings for movies. Predicting the unknown
matrix entry, which relates to user-item ratings, would result in a new
suggestion of a movie, item, or product to the user.

The usage of recent technology in recommendation systems has been
documented in . provides a thorough review of some highly notable works
on deep learning-based recommender systems. This paper served as
motivation to delve into Neural Architecture and Deep Learning. Deep
learning effectively captures the non-linear and non-trivial user-item
relationships, and enables the codification of more complex abstractions
as data representations in the higher layers, hence gaining popularity
over conventional recommendation systems.

Matrix completion techniques are explored in this project to build a
recommendation system. The method was implemented on the MovieLens
dataset using Keras to create a flexible model that can predict the
possible ratings the user can give to unwatched movies and suggest the
best among them.