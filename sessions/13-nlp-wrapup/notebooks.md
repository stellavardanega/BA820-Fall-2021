# Wrap Up - State of NLP and Misc

1. Unsupervised Topic Modeling with all the lego peices we saw in class!
2. Conversational AI
3. Embeddings are where everyone's attention is -- deep learning approaches
4. Finetuning transformer models
5. Putting it all together - Kaggle like challenge

# 1. Topic Modeling

[Topic Modeling - Top2Vec](https://colab.research.google.com/drive/18zD1I_WYRw_DmXVqaUQ4by7GbOPCabGN?usp=sharing)

- https://arxiv.org/abs/2008.09470
- https://github.com/ddangelov/Top2Vec

### Other Considerations for Topic Modeling

My issue is that classical Topic Modeling doesn't really work well, at least in my opinion.  It's very hard to get it to produce meaningful output, but many attempt to apply this approach.  Doesn't work well for short text.  Document is a mixture of topics, and topics are a mixture of words.

- https://radimrehurek.com/gensim/models/ldamodel.html
- [approachable discussion](https://highdemandskills.com/topic-modeling-intuitive/)

There are newer, deep learning approaches that you might consider:

Below considers the user of BERT transformers to help identify topics.  
- https://github.com/MaartenGr/BERTopic


# 2.  Rasa - Conversational AI

- [rasa](https://rasa.com/)
- [rasa learning](https://learning.rasa.com/)
- [rasa docs and research](https://rasa.com/docs/)
- [rasa playground](https://rasa.com/docs/rasa/playground/)
- [rasa examples and repos](https://github.com/orgs/RasaHQ/repositories)

> Rasa combines tools like spacy and scikit-learn under the hood to build out intent classification and entity recognition models to help drive machine-learning backed conversations.

### Alternatives and Considrations

- [Google Cloud and Diagflow](https://developers.google.com/learn/topics/chatbots)

Worth noting that chatbots have gone horribly wrong in the past.

- (Microsoft Tay gone wrong)[https://spectrum.ieee.org/in-2016-microsofts-racist-chatbot-revealed-the-dangers-of-online-conversation]



# 3.  Embeddings are Everywhere 

[Universal Sentence Embeddings with spacy](https://colab.research.google.com/drive/1Uvdr09Aitq4IQHTj5Yb-WkoUWE4g_d5X?usp=sharing)


[`finetune`  deep learning transformer models](https://colab.research.google.com/drive/1jXd2VZdDjudEj25Ij7-E7N2Qa5IghHQ8?usp=sharing)


### Additional Considerations and Resources

- [BERT visually explained - networks are part of SOTA](https://jalammar.github.io/illustrated-bert/)
- [finetune](https://github.com/IndicoDataSolutions/finetune)

HuggingFace is the coolest kid on the block right now.  They are doing some amazing work to make transformers approachable and easy to plug into a project.

- [HuggingFace](https://huggingface.co/)
- [NLG](https://transformer.huggingface.co/)
- [Course for pytorch or tensorflow](https://huggingface.co/course/chapter1/1?fw=tf)


# 4.  Data Challenge

Put it all together!  We saw that sometimes our "standard" ML models do well and are fast to fit/predict, but see how well you can do applying the concepts from class to predict the customer intent!



