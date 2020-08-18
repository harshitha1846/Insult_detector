# Insult_detector
It is a supervised machine learning model which detects whether a comment is insult or not.
## Why Insult Detector?
In the present days people are using various online forums, blogs, social
networking sites, newsgroups as source of their networking, sharing and receiving
knowledge. Sometimes, some users may make comments that are considered as unpleasant
by other users and may hurt their feelings. Besides, it is a barrier to user participant and
prevents new users from participating. Also, sometimes you are looking for some information
on some site and find insults then it leads to frustration. So, to avoid such incidents a proper
interface that screens the offensive comments before posting in the social media platforms is
very much needed. Also, it is not possible to have a human moderator to review the
comments before posting because of the increasing amount of online data. Hence, we need an
automatic classifier that will detect the insulting comments. 
## Goal of this project
Our aim is to focus on comments
that are insulting to other participants of the blog/ forum conversation. However, the
comments containing insults but are targeted to a non-participant of conversation (like a
celebrity etc.) are not marked as insults. Insults are of many types like: Taunts, squalid
language, slurs and racism which are aimed at attacking the other person. However, some
insults are aimed at abuse or embarrassing the reader (not an attack) like crude language,
disguise provocative words, disguise, sarcasm, innuendo (indirect reference). We are aimed
at detecting comments that are intended to be obviously insulting to other participants and
when the comment is detected to be an insult then the user is not allowed to comment, else
the user is allowed to post the comment without any restriction
## How?
We obtained our annotated dataset from the social media websites. The training dataset
consists of nearly 5000 comments which are labelled as positive and negative. This problem is
treated as binary classification problem and using various machine learning algorithms like Logistic
regression, SVM have been used in the model.But before that we have done some preprocessing techniques(nlp) on text such that it is feasible to feed to algorithm.
## Reqirements
 - flask
 - sklearn
 - nltk
 - numpy
 
