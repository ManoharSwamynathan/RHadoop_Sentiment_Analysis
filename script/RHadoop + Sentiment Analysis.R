# This is based on the tutorials of following links
# https://sites.google.com/site/miningtwitter/questions/sentiment/sentiment
# http://jeffreybreen.wordpress.com/2011/07/04/twitter-text-mining-r-slides/
# https://github.com/benmarwick/AAA2011-Tweets/blob/master/AAA2011.R
#
# Requires rmr2 package (https://github.com/RevolutionAnalytics/RHadoop/wiki).

# Load libraries
library(rmr2)
library(plyr)

# Set up the evironment variablnes
# The path may change depending on your verion of hadoop installation
Sys.setenv("HADOOP_CMD"="/usr/local/hadoop/bin/hadoop"
Sys.setenv("HADOOP_STREAMING"="/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.4.0.jar")

# Root folder path
setwd('/home/manohar/example/Sentiment_Analysis')

# Set "LOCAL" variable to T to execute using rmr's local backend.
# Otherwise, use Hadoop (which needs to be running, correctly configured, etc.)
LOCAL=F

if (LOCAL)
{
  rmr.options(backend = 'local')
  
  # we have smaller extracts of the data in this project's 'local' subdirectory
  hdfs.data.root = '/home/manohar/example/Sentiment_Analysis/'
  hdfs.data = file.path(hdfs.data.root, 'data', 'data.csv')
  
  hdfs.out.root = hdfs.data.root
  
} else {
  rmr.options(backend = 'hadoop')
  
  # assumes 'Sentiment_Analysis/data' input path exists on HDFS under /home/manohar/example
  
  hdfs.data.root = '/home/manohar/example/Sentiment_Analysis/'
  hdfs.data = file.path(hdfs.data.root, 'data')
  
  # writes output to 'Sentiment_Analysis' directory in user's HDFS home (e.g., /home/manohar/example/Sentiment_Analysis/)
  hdfs.out.root = 'Sentiment_Analysis'
}

hdfs.out = file.path(hdfs.out.root, 'out')

# equivalent to hadoop dfs -copyFromLocal
# will copy file from local to hadoop, if already exists then will return TRUE
hdfs.put(hdfs.data,  hdfs.data)

# asa.csv.input.format() - read CSV data files and label field names
# for better code readability (especially in the mapper)
#
asa.csv.input.format = make.input.format(format='csv', mode='text', streaming.format = NULL, sep=',',
                                         col.names = c('ID', 'Name', 'Gender', 'Age','OverAllRating',                                             
                                                       'ReviewType', 'ReviewTitle', 'Benefits', 'Money', 'Experience', 
                                                       'Purchase', 'claimsProcess', 'SpeedResolution', 'Fairness',            
                                                       'ReviewDate', 'Review', 'Recommend', 'ColCount'),
                                         stringsAsFactors=F)


# load opinion lexicons
pos_words <- scan('/home/manohar/example/Sentiment_Analysis/data/positive-words.txt', what='character',     comment.char=';')
neg_words <- scan('/home/manohar/example/Sentiment_Analysis/data/negative-words.txt', what='character', comment.char=';')

# sentiment analysis
# source: https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107
score.sentiment = function(sentence, pos.words, neg.words)
{
  require(plyr)
  require(stringr)
  
  score = laply(sentence, function(sentence, pos.words, neg.words) {
    # clean up sentences with R's regex-driven global substitute, gsub():
    sentence = gsub('[[:punct:]]', '', sentence)
    sentence = gsub('[[:cntrl:]]', '', sentence)
    sentence = gsub('\\d+', '', sentence)
    # and convert to lower case:
    sentence = tolower(sentence)
    
    # split into words. str_split is in the stringr package
    word.list = str_split(sentence, '\\s+')
    # sometimes a list() is one level of hierarchy too much
    words = unlist(word.list)
    
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    
    # match() returns the position of the matched term or NA
    # we just want a TRUE/FALSE:
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
    score = sum(pos.matches) - sum(neg.matches)
    
    return(score)
  }, pos.words, neg.words)
  
  score.df = data.frame(score)
  return(score.df)
}

# the mapper gets keys and values from the input formatter
# in our case, the key is NULL and the value is a data.frame from read.table()

mapper = function(key, val.df) {  
  # Remove header lines
  val.df = subset(val.df, Review != 'Review')
  output.key = data.frame(Review = as.character(val.df$Review),stringsAsFactors=F)
  output.val = data.frame(val.df$Review)
  return( keyval(output.key, output.val) )
}

reducer = function(key, val.df) {  
  output.key = key
  output.val = data.frame(score.sentiment(val.df, pos_words, neg_words))
  return( keyval(output.key, output.val) )  
}


mr.sa = function (input, output) {
  mapreduce(input = input,
            output = output,
            input.format = asa.csv.input.format,
            map = mapper,
            reduce = reducer,
            verbose=T)
}

out = mr.sa(hdfs.data, hdfs.out)
results = from.dfs(out)
# put the result in a dataframe
df = sapply(results,c)
df = data.frame(df)
colnames(df) <- c('Review', 'score')

print(head(df))
str(df)

library(plyr)
library(ggplot2)
library(grid)
ggplot(df, aes(x=score)) + 
  geom_histogram(binwidth=1) + 
  xlab("Sentiment score") + 
  ylab("Frequency") + 
  theme_bw()  + 
  theme(axis.title.x = element_text(vjust = -0.5, size = 14)) + 
  theme(axis.title.y=element_text(size = 14, angle=90, vjust = -0.25)) + 
  theme(plot.margin = unit(c(1,1,2,2), "lines"))

review.pos<- subset(df,df$score>= 2) 
review.neg<- subset(df,df$score<= -2)
claim <- subset(df, regexpr("claim", df$Review) > 0) 
ggplot(claim, aes(x = score)) + geom_histogram(binwidth = 1) + xlab("Sentiment score for the token 'claim'") + ylab("Frequency") + theme_bw()  + theme(axis.title.x = element_text(vjust = -0.5, size = 14)) + theme(axis.title.y = element_text(size = 14, angle = 90, vjust = -0.25)) + theme(plot.margin = unit(c(1,1,2,2), "lines"))

#classify emotion
class_emo = classify_emotion(df$Review, algorithm="bayes", prior=1.0)
#get emotion best fit
emotion = class_emo[,7]
# substitute NA's by "unknown"
emotion[is.na(emotion)] = "unknown"

# classify polarity
class_pol = classify_polarity(df$Review, algorithm="bayes")

# get polarity best fit
polarity = class_pol[,4]

# data frame with results
sent_df = data.frame(text=df$Review, emotion=emotion, polarity=polarity, stringsAsFactors=FALSE)

# sort data frame
sent_df = within(sent_df, emotion <- factor(emotion, levels=names(sort(table(emotion), decreasing=TRUE))))

# plot distribution of emotions
ggplot(sent_df, aes(x=emotion)) +
  geom_bar(aes(y=..count.., fill=emotion)) +
  scale_fill_brewer(palette="Dark2") +
  labs(x="emotion categories", y="number of Feedback", 
       title = "Sentiment Analysis of Feedback about claim(classification by emotion)",
       plot.title = element_text(size=12))

# plot distribution of emotions
ggplot(sent_df, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill=polarity)) +
  scale_fill_brewer(palette="RdGy") +
  labs(x="emotion categories", y="number of Feedback", 
       title = "Sentiment Analysis of Feedback about claim(classification by emotion)",
       plot.title = element_text(size=12))

# separating text by emotion
emos = levels(factor(sent_df$emotion))
nemo = length(emos)
emo.docs = rep("", nemo)
for (i in 1:nemo)
{
  tmp = df$Review[emotion == emos[i]]
  emo.docs[i] = paste(tmp, collapse=" ")
}

# remove stopwords
emo.docs = removeWords(emo.docs, stopwords("english"))
# create corpus
corpus = Corpus(VectorSource(emo.docs))
tdm = TermDocumentMatrix(corpus)
tdm = as.matrix(tdm)
colnames(tdm) = emos

# comparison word cloud
comparison.cloud(tdm, colors = brewer.pal(nemo, "Dark2"),
                 scale = c(3,.5), random.order = FALSE, title.size = 1.5)

