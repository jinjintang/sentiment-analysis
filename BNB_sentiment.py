import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.text import TextCollection
import numpy as np
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score
import sys
from sklearn.naive_bayes import BernoulliNB
nltk.download('stopwords')
def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)
def clean_data(data):
    data['text']=data['text'].str.lower()
    data['text']=data['text'].apply(lambda x:remove_urls(x))    
    emoticons_str = r"""
        (?:
            [:=;] # 眼睛
            [oO\-]? # ⿐⼦
            [D\)\]\(\]/\\OpP] # 嘴
        )"""
    regex_str = [
        emoticons_str,
        r'<[^>]+>', # HTML tags
        r'(?:@[\w_]+)', # @某⼈
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # 话题标签
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
                       # URLs
        r'(?:(?:\d+,?)+(?:\.?\d+)?)', # 数字
        r"(?:[a-z][a-z'\-_]+[a-z])", # 含有 - 和 ‘ 的单词
        r'(?:[\w_]+)', # 其他
        r'(?:\S)' # 其他
    ]

    tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
    emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
      
    def tokenize(s):
        return tokens_re.findall(s)
      
    def preprocess(s, lowercase=False):
        tokens = tokenize(s)
        if lowercase:
            tokens = [token if emoticon_re.search(token) else token.lower() for token in
    tokens]
        return tokens     


    data['words']=data['text'].apply(lambda x:preprocess(x))


    def remove_punctuation(line):
         #, @, , $ or %
        rule = re.compile(r"[^a-zA-Z0-9\#\@\_\$\%]")
        line = rule.sub('',line)
        return line

    def process(word_list):
        res=[]
        for word in word_list:
            word=remove_punctuation(word)
            if word=='':
                continue
            res.append(word)
        return res

    data['words']=data['words'].apply(lambda x:process(x))

    
    stop_words=stopwords.words('english')


    data['words']=data['words'].apply(lambda x:[ i for i in x if i not in stop_words])

    stemmer = nltk.stem.PorterStemmer()
    data['words']=data['words'].apply(lambda x:[ stemmer.stem(i) for i in x ])

    return data



if __name__=='__main__':
   
    dataset_path=sys.argv[1]
    testset_path=sys.argv[2]
    train=pd.read_csv(dataset_path,sep='\t',header=None,names=['number','text','topic','sentiment','sacastic'])
    #train=train[train['sentiment']!='neutral']
    test=pd.read_csv(testset_path,sep='\t',header=None,names=['number','text','topic','sentiment','sacastic'])
    data_all=pd.concat([train,test],axis=0)
    data_all=clean_data(data_all)
 
    words=[]
    for t in data_all['words']:
        words.extend(t)
    word_training=words
   
    count = CountVectorizer()
    count.fit(word_training)
    
    data_all['sent']=data_all['words'].apply(lambda x:' '.join(x))
    bag_of_words=count.transform(data_all['sent'].to_list()).toarray()
   
    #corpus=TextCollection(data_all['words'].to_list())
    #data_all['tfidf']=data_all['words'].apply(lambda x:np.array([corpus.tf_idf(word,x) for word in word_training]))
   
    bnb_sentiment= BernoulliNB(fit_prior=True) 
    
    train=data_all[:len(train)]
    test=data_all[len(train):]

    bnb_sentiment.fit(bag_of_words[:len(train)],np.array(train['sentiment']))

    #bnb_sentiment.fit(np.array(train['tfidf'].to_list()),np.array(train['sentiment']).reshape(-1,1))
    y_pred=bnb_sentiment.predict(bag_of_words[:len(train)])
    y_label=np.array(train['sentiment']).reshape(-1,1)
    #y_pred=bnb_sentiment.predict(np.array(train['tfidf'].to_list()))
    '''
    print('accuracy:',accuracy_score(y_label,y_pred))
    print('precision_micro:',precision_score(y_label,y_pred,average='micro'))
    print('precision_macro:',precision_score(y_label,y_pred,average='macro'))
    print('recall_micro:',recall_score(y_label,y_pred,average='micro'))
    print('recall_macro:',recall_score(y_label,y_pred,average='macro'))  
    print('f1_micro:',f1_score(y_label,y_pred, average='micro'))
    print('f1_macro:',f1_score(y_label,y_pred, average='macro'))

    print(classification_report(y_label,y_pred))
    '''
    y_pred=bnb_sentiment.predict(bag_of_words[len(train):])
        
    for i in range(len(y_pred)):
        print(test.iloc[i]['number'],y_pred[i])
    
    #y_pred=bnb_sentiment.predict(np.array(test['tfidf'].to_list()))
    



    
