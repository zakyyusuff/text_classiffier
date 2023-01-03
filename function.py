import pandas as pd
from pandas import DataFrame
import nltk

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

def get_feature(text):
    if len(text)==2:
        return {'ham':text[-3]}
    elif len(text)>=1:
        return {'spam':text[-3]}
    else:
        return {'ham':'', 'spam':''}
      
 def get_feature_text(text):
    if len(text)==2:
        return {'spam':'', 'ham':text[-1]}
    else:
        return {'spam':DataFrame.rename(text[-2])[0], 'ham':DataFrame.rename(text[-1])[0]}

def get_data(df, get_feature=get_feature):
    featrues = []
    for k, row in df.iterrows():
        text = row['v1']; type = row['v2']
        if isinstance(text, str):
            if ' ' in text:
                text = text.replace(' ', '')
            if '(' not in text:
                featrues.append((get_feature(text), type.strip('() ')))
            else:
                text = text.partition('(')[0]
                featrues.append((get_feature(text), type.strip('() ')))
    return featrues
  
def get_train_test(featrues, ratio=0.9):
    N = len(featrues)
    T = int(N * ratio)
    train = featrues[:T]
    test = featrues[T:]
    return train, test

def text_classifier(df, f=get_feature):
    data = get_data(df, f)
    train, test = get_train_test(data)
    classifier = nltk.NaiveBayesClassifier.train(train)
    acc = nltk.classify.accuracy(classifier, test)
    return classifier, acc

def show_type_of_text(text, texts=False, show_acc=False):
    f = get_feature_text if texts else get_feature
    classifier, acc = text_classifier(df, f)
    if show_acc:
        print(f'Accuracy: {acc:.4}') 
    clf = classifier.classify(f(text))
    print(f'{text}: {clf}')
    classifier.show_most_informative_features(10)

def give_type(type1='spam', type2='ham'): 
    data = get_data(df, get_feature)
    classifier = nltk.NaiveBayesClassifier.train(data)
    following = classifier.prob_classify({'ham':type2, 'spam':type1})
    x = following.generate()
    print(f'{type2}: {type1}{x}')

if __name__ == '__main__':
    print('-wait a minute-')
    show_type_of_text("spam")
    print('explain: (give text type and first word)')    
    give_type(type1='Go until jurong point, crazy..')   
    
