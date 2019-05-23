import pandas as pd 
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn import tree
from gensim.models import Word2Vec
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
import unicodedata
from sklearn import feature_extraction
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import time
import re
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score,accuracy_score
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

DATA_PATH = 'fake-news-detection/data.csv'
WORD2VEC_FILEPATH = 'glove.6B.50d.txt'
DATA_PATH_FNC = 'fnc-1-master/' 

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------


def calculateFNC(orig,pred):
    #cm = confusion_matrix(orig,pred)
    def score_submission(gold_labels, test_labels):
        score = 0.0
        cm = [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]

        for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
            g_stance, t_stance = g, t
            if g_stance == t_stance:
                score += 0.25
                if g_stance != 0:
                    score += 0.50
            if g_stance > 0 and t_stance > 0:
                score += 0.25
        return score
    res = score_submission(orig,pred)*1.0/score_submission(orig,orig)
    score = 0.0
    related = 0
    for i,j in zip(orig,pred):
        related += 1 if i != 0 else 0
        if i==0 and j==0 or i!=0 and j!=0:
            score += 0.25
        if i!=0 and i==j:
            score += 0.75
    resMine = round(score/ ( 0.25*len(orig) + 0.75*(related) ),3)
    # print "FNC->",res,resMine
    return res


def logisticRegression(D,L):
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=300)
    pred = cross_val_predict(clf, D, L, cv=10)
    clf.fit(D,L)
    return clf,pred

def neuralNetwork(D,L):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3), random_state=1)
    pred = cross_val_predict(clf, D, L, cv=10)
    clf.fit(D,L)
    return clf,pred
    
def randomForest(D,L):
    temp = {}
    for i in L:
        temp[i] = 1 if i not in temp else 1+temp[i]
    for i in temp:
        temp[i] = 1.0/temp[i]

    clf = RandomForestClassifier(n_estimators=20, max_depth=3,random_state=0,class_weight=temp)
    pred = cross_val_predict(clf, D, L, cv=10)
    clf.fit(D,L)
    return clf,pred

def decisionTree(D,L):
    clf = tree.DecisionTreeClassifier()
    pred = cross_val_predict(clf, D, L, cv=10)
    clf.fit(D,L)
    return clf,pred

def gradientBoosting(D,L):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    pred = cross_val_predict(clf, D, L, cv=10)
    clf.fit(D,L)
    return clf,pred

# ------------------------------------------------------------------------------------------------------------

class Word2Vec:
    def __init__(self,Path):
        data = open(Path).read().split('\n')
        data = map(lambda x: x.split(' '),data)
        
        data = filter(lambda x: len(x)==51,data)
        self.word2vec = {}
        for i in data:
            self.word2vec[i[0]] = map(lambda x: float(x), i[1:])
        
    
    def getVec(self,word):
        return self.word2vec[word] if word in self.word2vec else [0.0 for i in range(50)]


def getWordLenDist(tokens):
    MAX_WORD_LEN = 30
    tokenLen = map(lambda x: len(x),tokens)
    wordLenFeat = [0 for i in range(MAX_WORD_LEN)]
    for i,word in zip(tokenLen,tokens):
        if i<MAX_WORD_LEN:
            wordLenFeat[i] += 1
    tot = np.array(wordLenFeat).sum()
    for i in range(len(wordLenFeat)):
        
        wordLenFeat[i] /= tot*1.0
    return wordLenFeat

def getPosDist(tokens):
    tags = nltk.pos_tag(tokens)
    allTags = {"CC":0,"CD":1,"DT":2,"EX":3,"FW":4,"IN":5,"JJ":6,"JJR":7,"JJS":8,"LS":9,"MD":10,"NN":11,"NNS":12,"NNP":13,"NNPS":14,"PDT":15, \
    "POS":16,"PRP":17,"PRP$":18,"RB":19,"RBR":20,"RBS":21,"RP":22,"TO":23,"UH":24,"VB":25,"VBD":26,"VBG":27,"VBN":28,"VBP":29,"VBZ":30,"WDT":31, \
    "WP":32,"WP$":33,"WRB":34}

    feat = [0 for i in allTags]
    for word,tag in tags:
        if tag in allTags:
            feat[allTags[tag]] += 1
    
    tot = np.array(feat).sum()
    for i in range(len(feat)):
        feat[i] /= tot*1.0
    return feat

def getFeaturesFromText(text):
    tokens = nltk.word_tokenize(text)
    feat = [(len(text)-len(tokens))*1.0/len(tokens)]
    feat += getWordLenDist(tokens)
    # feat += getPosDist(tokens)
    # print feat  
    return feat

def getPunctuationFeatures(text):
    puncList = "~!@#$%^&*()-_|;:'\",./?"
    
    dic = { i:0 for i in puncList}
    for i in text:
        if i in dic:
            dic[i] += 1
    res = [dic[i]*1.0/len(text) for i in sorted(dic.keys()) ]
    return res

def readData(DATA_PATH):
    df = pd.read_csv(DATA_PATH,encoding="utf-8")#[:100]
    df['Headline'] = df['Headline'].apply(lambda val: unicodedata.normalize('NFKD', unicode(val)).encode('ascii', 'ignore').decode())
    df['Body'] = df['Body'].apply(lambda val: unicodedata.normalize('NFKD', unicode(val)).encode('ascii', 'ignore').decode())
    df['LengthHeadline'] = [len(headline) for headline in df['Headline']]
    df['LengthBody'] = [len(body) for body in df['Body']]
    return df

def readFNCData(PATH,fileType):
    dfStances = pd.read_csv(PATH+fileType+'_stances.csv',encoding="utf-8")
    dfBodies = pd.read_csv(PATH+fileType+'_bodies.csv',encoding="utf-8")
    bodyDic = {}
    for ind in range(len(dfBodies)):
        bodyDic[dfBodies['Body ID'].iloc[ind]] = dfBodies['articleBody'].iloc[ind]
    dfStances['Body'] = dfStances['Body ID'].apply(lambda val: bodyDic[val])
    dfStances['Label'] = dfStances['Stance'].apply(lambda x: {'unrelated': 0,'agree':1,'disagree':2, 'discuss':3 }[x])
    df = dfStances[['Headline','Body','Label']]
    df['LengthHeadline'] = [len(headline) for headline in df['Headline']]
    df['LengthBody'] = [len(body) for body in df['Body']]
     
     # print bodyDic.keys()
    return df


def getMostOccuringWordFeatures(text,n,word2vecObj):
    tokens = nltk.word_tokenize(text)
    tokens = filter(lambda x: x not in stopwords.words('english'),tokens)
    mcWord = nltk.FreqDist(tokens).most_common(n)
    
    res = np.array([0]*50,dtype=np.float64)
    for i in mcWord:    
        res += np.array(word2vecObj.getVec(i[0]))
    
    res /= n
    
    return list(res)
class WordDisFakeReal:
    def __init__(self,df):

        self.NUMBER_OF_WORDS = 25
        real_text = ' '.join(df[df['Label'] == 1]['Headline'])
        fake_text = ' '.join(df[df['Label'] == 0]['Headline'])
        fake_words = [word for word in nltk.tokenize.word_tokenize(fake_text) if word not in stopwords.words('english') and len(word) > 3]
        real_words = [word for word in nltk.tokenize.word_tokenize(real_text) if word not in stopwords.words('english') and len(word) > 3]
        common_fake = nltk.FreqDist(fake_words).most_common(self.NUMBER_OF_WORDS)
        common_real = nltk.FreqDist(real_words).most_common(self.NUMBER_OF_WORDS)
        self.dic = {}
        for i,j in common_fake+common_real:
            if i not in self.dic:
                self.dic[i] = True
        
        # fake_ranks = []
        # fake_counts = []
        # real_ranks = []
        # real_counts = []

        # for ii, word in enumerate(reversed(common_fake)):
        #     fake_ranks.append(ii)
        #     fake_counts.append(word[1])

        # for ii, word in enumerate(reversed(common_real)):
        #     real_ranks.append(ii)
        #     real_counts.append(word[1])
        
        # plt.figure(figsize=(20, 7))
        # plt.scatter(fake_ranks, fake_counts)
        # plt.scatter(real_ranks, real_counts)
        # plt.show()


    def getWordRankFeatures(self,text):
        count = {}
        for i in self.dic:
            count[i] = 0
        # print "GET WORD->",len(count),len(self.dic)
        toks = nltk.word_tokenize(text)
        for i in toks:
            if i in count:
                count[i] += 1
        res =  [count[i]*1.0/len(toks) for i in sorted(count.keys())]
        while(len(res) < 2*self.NUMBER_OF_WORDS):
            res += [0.0]
        # print "GET WORD RANK->",len(res)
        return res


def getFNCBaseLineFeatures(headline,body):
    _wnl = nltk.WordNetLemmatizer()
    def normalize_word(w):
        return _wnl.lemmatize(w).lower()
    def get_tokenized_lemmas(s):
        return [normalize_word(t) for t in nltk.word_tokenize(s)]
    def clean(s):
        return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()
    def remove_stopwords(l):
        # Removes stopwords from a list of tokens
        return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]
    def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
        if not os.path.isfile(feature_file):
            feats = feat_fn(headlines, bodies)
            np.save(feature_file, feats)
        return np.load(feature_file)
    def word_overlap_features(headlines, bodies):
        X = []
        for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
            clean_headline = clean(headline)
            clean_body = clean(body)
            clean_headline = get_tokenized_lemmas(clean_headline)
            clean_body = get_tokenized_lemmas(clean_body)
            features = [
                len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
            X.append(features)
        return X
    def refuting_features(headlines, bodies):
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            # 'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]
        X = []
        for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
            clean_headline = clean(headline)
            clean_headline = get_tokenized_lemmas(clean_headline)
            features = [1 if word in clean_headline else 0 for word in _refuting_words]
            X.append(features)
        return X
    def polarity_features(headlines, bodies):
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

        def calculate_polarity(text):
            tokens = get_tokenized_lemmas(text)
            return sum([t in _refuting_words for t in tokens]) % 2
        X = []
        for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
            clean_headline = clean(headline)
            clean_body = clean(body)
            features = []
            features.append(calculate_polarity(clean_headline))
            features.append(calculate_polarity(clean_body))
            X.append(features)
        return np.array(X)
    def ngrams(input, n):
        input = input.split(' ')
        output = []
        for i in range(len(input) - n + 1):
            output.append(input[i:i + n])
        return output
    def chargrams(input, n):
        output = []
        for i in range(len(input) - n + 1):
            output.append(input[i:i + n])
        return output
    def append_chargrams(features, text_headline, text_body, size):
        grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
        grams_hits = 0
        grams_early_hits = 0
        grams_first_hits = 0
        for gram in grams:
            if gram in text_body:
                grams_hits += 1
            if gram in text_body[:255]:
                grams_early_hits += 1
            if gram in text_body[:100]:
                grams_first_hits += 1
        features.append(grams_hits)
        features.append(grams_early_hits)
        features.append(grams_first_hits)
        return features
    def append_ngrams(features, text_headline, text_body, size):
        grams = [' '.join(x) for x in ngrams(text_headline, size)]
        grams_hits = 0
        grams_early_hits = 0
        for gram in grams:
            if gram in text_body:
                grams_hits += 1
            if gram in text_body[:255]:
                grams_early_hits += 1
        features.append(grams_hits)
        features.append(grams_early_hits)
        return features
    def hand_features(headlines, bodies):
        def binary_co_occurence(headline, body):
            # Count how many times a token in the title
            # appears in the body text.
            bin_count = 0
            bin_count_early = 0
            for headline_token in clean(headline).split(" "):
                if headline_token in clean(body):
                    bin_count += 1
                if headline_token in clean(body)[:255]:
                    bin_count_early += 1
            return [bin_count, bin_count_early]
        def binary_co_occurence_stops(headline, body):
            # Count how many times a token in the title
            # appears in the body text. Stopwords in the title
            # are ignored.
            bin_count = 0
            bin_count_early = 0
            for headline_token in remove_stopwords(clean(headline).split(" ")):
                if headline_token in clean(body):
                    bin_count += 1
                    bin_count_early += 1
            return [bin_count, bin_count_early]
        def count_grams(headline, body):
            # Count how many times an n-gram of the title
            # appears in the entire body, and intro paragraph

            clean_body = clean(body)
            clean_headline = clean(headline)
            features = []
            features = append_chargrams(features, clean_headline, clean_body, 2)
            features = append_chargrams(features, clean_headline, clean_body, 8)
            features = append_chargrams(features, clean_headline, clean_body, 4)
            features = append_chargrams(features, clean_headline, clean_body, 16)
            features = append_ngrams(features, clean_headline, clean_body, 2)
            features = append_ngrams(features, clean_headline, clean_body, 3)
            features = append_ngrams(features, clean_headline, clean_body, 4)
            features = append_ngrams(features, clean_headline, clean_body, 5)
            features = append_ngrams(features, clean_headline, clean_body, 6)
            return features

        X = binary_co_occurence(headline, body) + binary_co_occurence_stops(headline, body) + count_grams(headline, body)
        return X
    return hand_features(headline,body)



def createData(df,clf=None):       
    global WORD2VEC_FILEPATH
    data = []
    label = []

    obj = WordDisFakeReal(df)
    word2vecObj = Word2Vec(WORD2VEC_FILEPATH)
    
    print "\nData Preparation Started!"
    for i in range(len(df)):
        if (i+1)%500 == 0:
            print "Processed %d/%d" % (i+1,len(df))
        body = df['Body'].iloc[i]
        head = df['Headline'].iloc[i]
        try:
            locFeature = []
            # a = len(locFeature)
            locFeature += getFeaturesFromText(body) + getFeaturesFromText(head)

            # print "1->",len(locFeature)
            # a = len(locFeature)
            
            locFeature += [ df['LengthBody'].iloc[i] , df['LengthHeadline'].iloc[i] ]

            # print "2->",len(locFeature)
            # a = len(locFeature)
            
            locFeature += getPunctuationFeatures(body) + getPunctuationFeatures(head)

            # print "3->",len(locFeature)
            # a = len(locFeature)

            locFeature += obj.getWordRankFeatures(df['Headline'].iloc[i])+obj.getWordRankFeatures(df['Body'].iloc[i])

            # print "4->",len(locFeature)
            # a = len(locFeature)

            locFeature += getMostOccuringWordFeatures(body,15,word2vecObj) + getMostOccuringWordFeatures(head,3,word2vecObj)

            # print "5->",len(locFeature)
            # a = len(locFeature)

            locFeature += getFNCBaseLineFeatures(head,body)

            # print "6->",len(locFeature)
            # a = len(locFeature)

            if clf is not None:
                locFeature += clf.predict([locFeature])
                # print clf.predict([locFeature])
            data.append(locFeature)
            label.append(df['Label'].iloc[i])
        except Exception as e:
            print "-------- ERROR:",e
            pass
            # print '--------- ERROR ----------'
            # print head
            # print body
    print "Data Preparation Complete!\n---------------------------\n"
    return data,label        



# ------------------------------------------------------------------------------------------------------------

def printScore(alg,orig,pred,FNC=False):
    fncScore = calculateFNC(orig,pred) if FNC else 0.0 
    print "\t%25s ->  F1: %.3f  Acc: %.3f  FNC: %.3f" % (alg,f1_score(orig,pred,average='macro'),accuracy_score(orig,pred),fncScore)


def runAllAlgorithm(header,dataTr,labelTr,dataTe,labelTe,fnc=False):
    clfDT,predDT = decisionTree(dataTr,labelTr)
    clfLR,predLR = logisticRegression(dataTr,labelTr)
    clfGB,predGB= gradientBoosting(dataTr,labelTr)
    clfRF,predRF = randomForest(dataTr,labelTr)
    clfNN,predNN = neuralNetwork(dataTr,labelTr)

    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print header
    printScore("CV Decision Tree",labelTr,predDT,fnc)
    printScore("CV Logistic Regression",labelTr,predLR,fnc)
    printScore("CV Gradient Boost",labelTr,predGB,fnc)
    printScore("CV Random Forest",labelTr,predRF,fnc)
    printScore("CV Neural Network",labelTr,predNN,fnc)
    print ""
    printScore("Decision Tree",labelTe,clfDT.predict(dataTe),fnc)
    printScore("Logistic Regression",labelTe,clfLR.predict(dataTe),fnc)
    printScore("Gradient Boost",labelTe,clfGB.predict(dataTe),fnc)
    printScore("Random Forest",labelTe,clfRF.predict(dataTe),fnc)
    printScore("Neural Network",labelTe,clfNN.predict(dataTe),fnc)
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    return clfDT


def getFeature(data,featureList):
    lst = [ (0,0) , (0,62) , (62,64) , (64,108) , (108,208) , (208,308) , (308,334)  ]

    new = []
    for i in data:
        row = []
        for j in featureList:
            start = lst[j][0]
            end = lst[j][1]
            row += i[start:end]
        new += [row]
    return new
        
    
def runDifferentFeatureCombination(data,label):
    comb = [[1], [2] , [3] , [4] , [5] , [6] , [1,6] , [2,6] , [3,6] , [4,6] , [5,6] , [2,3,4,5,6] , [1,3,4,5,6] , [1,2,4,5,6] ,[1,2,3,5,6], [1,2,3,4,6] , [1,2,3,4,5] , [1,2,3,4,5,6] ]
    trainSampleCount = int(0.8*len(data))
    
    for i in comb:
        sdata = getFeature(data,i)
        print "Comb->",i," LEngth->",len(sdata[0])
        runAllAlgorithm("\nRunning Feature Combinations "+str(i),sdata[:trainSampleCount],label[:trainSampleCount],sdata[trainSampleCount:],label[trainSampleCount:])




fullStart = time.time()

dfKaggle = readData(DATA_PATH)
print "Read kaggle data!"
# dfFNCComp = readFNCData(DATA_PATH_FNC,'competition_test')
# dfFNC = readFNCData(DATA_PATH_FNC,'train')


# dataFNCComp,labelFNCComp = createData(dfFNCComp)
# dataFNC,labelFNC = createData(dfFNC)


# clf = runAllAlgorithm("\nPerformance on Fake News Challenge (Competition Data)",dataFNC,labelFNC,dataFNCComp,labelFNCComp,True)

# temp = int(0.8*len(dataFNC))
# runAllAlgorithm("\nPerformance on Fake News Challenge (Train Data 80-20 split)",dataFNC[:temp],labelFNC[:temp],dataFNC[temp:],labelFNC[temp:],True)



dataKaggle,labelKaggle = createData(dfKaggle,None)


runDifferentFeatureCombination(dataKaggle,labelKaggle)
# trainSampleCount = int(0.8*len(dataKaggle))
# runAllAlgorithm("\nPerformance on Kaggle Data without Stance Feature",dataKaggle[:trainSampleCount],labelKaggle[:trainSampleCount],dataKaggle[trainSampleCount:],labelKaggle[trainSampleCount:])




# dataKaggle,labelKaggle = createData(dfKaggle,clf)
# trainSampleCount = int(0.8*len(dataKaggle))
# runAllAlgorithm("\nPerformance on Kaggle Data with Stance Feature",dataKaggle[:trainSampleCount],labelKaggle[:trainSampleCount],dataKaggle[trainSampleCount:],labelKaggle[trainSampleCount:])

