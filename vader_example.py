import pandas as pd
train=pd.read_csv('./dataset.tsv',sep='\t',header=None,names=['number','text','topic','sentiment','sacastic'])
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score

analyser = SentimentIntensityAnalyzer()
def func(x):
    d=analyser.polarity_scores(x)
    
    return d
score= train['text'].apply(lambda x:func(x))
score = score.apply(lambda x:sorted(x.items(), key=lambda x: x[1], reverse=True)[0][0])

score[(score!='pos')&(score!='neg')]='neural'
score[score=='pos']='postive'
score[score=='neg']='negative'
y_label=train['sentiment']
y_pred=score
print('accuracy:',accuracy_score(y_label,y_pred))
print('precision_micro:',precision_score(y_label,y_pred,average='micro'))
print('precision_macro:',precision_score(y_label,y_pred,average='macro'))
print('recall_micro:',recall_score(y_label,y_pred,average='micro'))
print('recall_macro:',recall_score(y_label,y_pred,average='macro'))  
print('f1_micro:',f1_score(y_label,y_pred, average='micro'))
print('f1_macro:',f1_score(y_label,y_pred, average='macro'))

print(classification_report(y_label,y_pred))

