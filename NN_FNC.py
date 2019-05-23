import torch
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torch import nn,optim
import unicodedata
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
from sklearn import feature_extraction
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
from scipy import spatial
import math
import sys
import re
import pickle

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

DATA_PATH = 'fake-news-detection/data.csv'
DATA_PATH_FNC = 'fnc-1-master/' 
WORD2VEC_FILEPATH = 'glove.6B.50d.txt'
HEAD_WORDS = 50
BODY_WORDS = 150
ExternalFeatureCount = 0

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

class Word2Vec:
    def __init__(self,Path):
        data = open(Path).read().split('\n')
        data = map(lambda x: x.split(' '),data)
        
        data = filter(lambda x: len(x)==51,data)
        self.word2vec = {}
        for i in data:
            self.word2vec[i[0]] = map(lambda x: float(x), i[1:])
        
    
    def getVec(self,word):
        # print "In word 2 vec:",word
        return self.word2vec[word] if word in self.word2vec else [0.0 for i in range(50)]

word2VecObjX = Word2Vec(WORD2VEC_FILEPATH)
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

def getMyFeatures(head,body):
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
           
    global word2VecObjX
    locFeature = []
    locFeature += getFeaturesFromText(body) + getFeaturesFromText(head)
    locFeature += [ len(body) , len(head) ]
    locFeature += getPunctuationFeatures(body) + getPunctuationFeatures(head)
    # locFeature += obj.getWordRankFeatures(head)+obj.getWordRankFeatures(body)
    locFeature += getMostOccuringWordFeatures(body,15,word2VecObjX) + getMostOccuringWordFeatures(head,3,word2VecObjX)
    locFeature += getFNCBaseLineFeatures(head,body)
    # print len(locFeature)
    return locFeature





# ------------------------------------------------------------------------------------------------------------


def readData(PATH):
    def labelToTensor(label,size):
        res = torch.zeros(size).long()
        res[label] = 1
        return res

    df = pd.read_csv(PATH,encoding="utf-8")#[:500] 
    df['Headline'] = df['Headline'].apply(lambda val: unicodedata.normalize('NFKD', unicode(val)).encode('ascii', 'ignore').decode())
    df['Body'] = df['Body'].apply(lambda val: unicodedata.normalize('NFKD', unicode(val)).encode('ascii', 'ignore').decode())
    df['Label'] = df['Label'].apply(lambda x: labelToTensor(x,2))
    return df

def readFNCData(PATH,fileType,problem):
    df = pd.read_csv(PATH+fileType+'_stances.csv',encoding="utf-8")
    dfBodies = pd.read_csv(PATH+fileType+'_bodies.csv',encoding="utf-8")
    bodyDic = {}
    headDic = {}
    for ind in range(len(dfBodies)):
        bodyDic[dfBodies['Body ID'].iloc[ind]] = dfBodies['articleBody'].iloc[ind]
    
    df['Body'] = df['Body ID'].apply(lambda x: bodyDic[x])
    label = {'unrelated':torch.tensor([1,0,0,0],dtype=torch.float) , 'agree':torch.tensor([0,1,0,0],dtype=torch.float) , 'disagree':torch.tensor([0,0,1,0],dtype=torch.float) , 'discuss':torch.tensor([0,0,0,1],dtype=torch.float)}
    
    if problem==1:
        label = {'unrelated':torch.tensor([1,0],dtype=torch.float) , 'agree':torch.tensor([0,1],dtype=torch.float) , 'disagree':torch.tensor([0,1],dtype=torch.float) , 'discuss':torch.tensor([0,1],dtype=torch.float)}
    elif problem==2:
        df = df.loc[df['Stance'] != 'unrelated']
        label = {'agree':torch.tensor([1,0,0],dtype=torch.float) , 'disagree':torch.tensor([0,1,0],dtype=torch.float) , 'discuss':torch.tensor([0,0,1],dtype=torch.float)}
        

    df['Label'] =  df['Stance'].apply(lambda val: label[val])
    print("Problem %d data read: %d " %(problem,len(df)))
    print "Read Data!"
    
    return df
    
def padList(lst,l,padder):
    while(len(lst)<l):
        lst += [padder]
    return lst

class FixedWidth(Dataset):
    def __init__(self,df,word2VecObj,dic,fileName):
        self.df = df
        self.w2v = word2VecObj
        self.res = {}
        self.weight = {}
        self.id = dic
        global ExternalFeatureCount
    
        try:
            f = open(fileName,'r')
            self.res = pickle.load(f)
            print "Object file found, loading from there instead, num features:",len(self.res[0][0]),len(self.res[0][1]),len(self.res[0][2])
            ExternalFeatureCount = len(self.res[0][2])
            return
        except:
            print "Object File Not Found"

        for idx in range(len(df)):
            if (idx+1)%300==0:
                print "Processed %d/%d" % (idx+1,len(df))
            label = self.df['Label'].iloc[idx]

            feat = getFNCBaseLineFeatures(self.df['Headline'].iloc[idx],self.df['Body'].iloc[idx]); ExternalFeatureCount = 26
            # feat = []; ExternalFeatureCount = 0
            # feat = getMyFeatures(self.df['Headline'].iloc[idx],self.df['Body'].iloc[idx]); ExternalFeatureCount = 234
            
            label = self.df['Label'].iloc[idx]
            head =  nltk.word_tokenize(self.df['Headline'].iloc[idx])
            body =  nltk.word_tokenize(self.df['Body'].iloc[idx])

            
            
            cosineInclude = True
            if cosineInclude:
                MCHead = nltk.FreqDist( filter(lambda x: x not in stopwords.words('english') , head) ).most_common(5)
                MCBody = nltk.FreqDist(filter(lambda x: x not in stopwords.words('english') , body)).most_common(15)
                MCHead = map(lambda x:x[0],MCHead)
                MCBody = map(lambda x:x[0],MCBody)

                
                cosSum = 0
                for i in MCHead:
                    for j in MCBody:
                        cos = spatial.distance.cosine(self.w2v.getVec(i),self.w2v.getVec(j))
                        cos = 0 if math.isnan(cos) else cos
                        cosSum += cos
                cosSum /= len(MCHead)*len(MCBody)
                ExternalFeatureCount += 1
                feat += [cosSum ]
            
            
            head = head[:HEAD_WORDS]
            body = body[:BODY_WORDS]
            head = padList(head,HEAD_WORDS,'!!___!!')            
            body = padList(body,BODY_WORDS,'!!___!!')

            head = map(lambda x: torch.tensor(self.w2v.getVec(x)),head)
            body = map(lambda x: torch.tensor(self.w2v.getVec(x)),body)
            
            head = torch.stack(head)
            body = torch.stack(body)
            
            
            self.res[idx] = head,body,torch.tensor(feat).float(),label
        
        f = open(fileName,'w')
        pickle.dump(self.res,f)
        print "Fixed width initialized"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        global HEAD_WORDS
        global BODY_WORDS
        
        
        
        # print "Data Loader Size:",head.size(),body.size(),label.size()
        return self.res[idx]

def getMostCommonWordsDic(df):
    temp = "".join(df['Headline'].values) + "".join(df['Body'])
    words = nltk.FreqDist(nltk.word_tokenize(temp)).most_common(BODY_WORDS+HEAD_WORDS)
    ident = {}
    for ind in range(len(words)):
        ident[words[ind][0]] = ind
    return ident

class HandCrafted(Dataset):
    def __init__(self,df,w2v,dic):
        self.df = df
        self.res = {}
        self.weight = {}
        self.w2v = w2v
        self.id = dic

        for idx in range(len(df)):
            if idx%300==0:
                print "Processed %d/%d" % (idx+1,len(df))
            label = self.df['Label'].iloc[idx]
            feat = getFNCBaseLineFeatures(self.df['Headline'].iloc[idx],self.df['Body'].iloc[idx])
            self.res[idx] = torch.tensor(feat),torch.tensor(feat),label
        print "initialized"
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        global HEAD_WORDS
        global BODY_WORDS
        
        
        
        # print len(feat)
        return self.res[idx]



class BagOfWord(Dataset):
    def __init__(self,df,w2v,dic):
        self.df = df
        self.res = {}
        self.weight = {}
        self.w2v = w2v
        self.id = dic

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        global HEAD_WORDS
        global BODY_WORDS
        
        
        label = self.df['Label'].iloc[idx]
        head =  nltk.word_tokenize(self.df['Headline'].iloc[idx])
        body =  nltk.word_tokenize(self.df['Body'].iloc[idx])

        headVector = [0.0 for i in self.id]
        bodyVector = [0.0 for i in self.id]

        for i in head:
            if i in self.id:
                headVector[self.id[i]] += 1
        
        for i in body:
            if i in self.id:
                bodyVector[self.id[i]] += 1
        
        # print headVector
        # print bodyVector
        cos = spatial.distance.cosine(headVector, bodyVector)
        # print "Cosine:",math.isnan(cos),cos
        cos = 0 if math.isnan(cos) else cos
        
        head = torch.tensor(headVector+[cos])
        body = torch.tensor(bodyVector)
        
        # print "Data Loader Size:",head.size(),body.size(),label.size()
        return head,body,label
class BagOfWordMostCommonW2V(Dataset):
    def __init__(self,df,w2v,id):
        self.df = df
        self.res = {}
        self.weight = {}
        self.w2v = w2v
        self.id = {}

        for idx in range(len(df)):
            label = self.df['Label'].iloc[idx]
            pattern = re.compile('[^\w\s]')
            
            head =  nltk.FreqDist(nltk.word_tokenize(re.sub(pattern, '', self.df['Headline'].iloc[idx]).lower() )).most_common(HEAD_WORDS)
            body =  nltk.FreqDist(nltk.word_tokenize( re.sub(pattern, '', self.df['Body'].iloc[idx]).lower() )).most_common(BODY_WORDS)

            # print head,body
            head = map(lambda x: x[0] , head)
            body = map(lambda x: x[0] , body)

            head = padList(head,HEAD_WORDS,'!!___!!')            
            body = padList(body,BODY_WORDS,'!!___!!')

            # print "Len HEAD -> ",len(head),len(filter(lambda x: x in self.w2v.word2vec,head))
            # print "Len BODY -> ",len(body),len(filter(lambda x: x in self.w2v.word2vec,body))
            head = map(lambda x: self.w2v.getVec(x.lower()),head)
            body = map(lambda x: self.w2v.getVec(x.lower()),body)
            
            cos = 0#spatial.distance.cosine(head, body)
            cos = 0 if math.isnan(cos) else cos 
            head = torch.tensor(head)
            body = torch.tensor(body)
            self.res[idx] =     head,body,label
        print "Bag of words most common initialized"
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        global HEAD_WORDS
        global BODY_WORDS
        
        
        
        # print head
        # print body
        
        # print "Cosine:",math.isnan(cos),cos
        
        
        # print "Data Loader Size:",head.size(),body.size(),label.size()
        return self.res[idx]

class BagOfWord(Dataset):
    def __init__(self,df,dic):
        self.df = df
        self.res = {}
        self.weight = {}

        self.id = dic

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        global HEAD_WORDS
        global BODY_WORDS
        
        
        label = self.df['Label'].iloc[idx]
        head =  nltk.word_tokenize(self.df['Headline'].iloc[idx])
        body =  nltk.word_tokenize(self.df['Body'].iloc[idx])

        headVector = [0.0 for i in self.id]
        bodyVector = [0.0 for i in self.id]

        for i in head:
            if i in self.id:
                headVector[self.id[i]] += 1
        
        for i in body:
            if i in self.id:
                bodyVector[self.id[i]] += 1
        
        # print headVector
        # print bodyVector
        cos = spatial.distance.cosine(headVector, bodyVector)
        # print "Cosine:",math.isnan(cos),cos
        cos = 0 if math.isnan(cos) else cos
        
        head = torch.tensor(headVector+[cos])
        body = torch.tensor(bodyVector)
        
        # print "Data Loader Size:",head.size(),body.size(),label.size()
        return head,body,label


class Cosine(Dataset):
    def __init__(self,df,word2vec,tfidfHead,tfidfBody):
        global HEAD_WORDS
        global BODY_WORDS

        self.res = {} 
        self.weight = {}
        self.l = len(df)
        
        tfHead = tfidfHead.transform(df['Headline'].values).toarray().tolist()
        tfBody = tfidfBody.transform(df['Body'].values).toarray().tolist()
        for idx in range(len(df)):
            if (idx+1)%100==0:
                print "Processed %d/%d" % (idx+1,len(df))
            cos = spatial.distance.cosine(tfHead[idx], tfBody[idx])
            # print "Cosine:",cos,math.isnan(cos)
            cos = 0 if math.isnan(cos) else cos
            self.res[idx] = torch.tensor(tfHead[idx]+tfBody[idx]+[cos]+getFNCBaseLineFeatures(df['Headline'].iloc[idx],df['Body'].iloc[idx])),df['Label'].iloc[idx]
            self.res[idx] = torch.tensor(  tfHead[idx]+tfBody[idx]+[cos] + getFNCBaseLineFeatures(df['Headline'].iloc[idx],df['Body'].iloc[idx])   ).float(),df['Label'].iloc[idx]

    def __len__(self):
        return self.l
    
    def __getitem__(self, idx):
        inp,label = self.res[idx]
        # print inp.size()
        # print label.size()
        # print inp.size(),inp
        return inp,label
    



# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers ,output_size,bidirectional,train,test):
#         super(RNN, self).__init__()

#         self.hidden_size = hidden_size
#         self.trainLoader = train
#         self.testLoader = test
#         self.bidirectional = bidirectional
#         self.Layer1 = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, bidirectional = self.bidirectional, nonlinearity = 'relu')
#         # self.Layer1 = nn.Linear(input_size,hidden_size)
#         self.Layer2 = nn.Linear(hidden_size*(2 if self.bidirectional else 1),output_size)

#     def forward(self, x):
#         # print "INPUTER->",type(x)
#         out,_ = self.Layer1(x)       
#         # print "out ",out.size()
#         x = self.Layer2(out[-1])
#         return x

class SimpleNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers , output_size, bidirectional ,datasetLoaderTrain,datasetLoaderTest):
        super(SimpleNN, self).__init__()
        self.trainLoader = datasetLoaderTrain
        self.testLoader = datasetLoaderTest
        global HEAD_WORDS
        global BODY_WORDS
        global ExternalFeatureCount
        self.LayerO = nn.Linear( ExternalFeatureCount  , hidden_size)
        self.Layer1 = nn.Linear((HEAD_WORDS + BODY_WORDS)*50  , hidden_size/5)
        self.Layer2 = nn.Dropout(0.3)
        self.Layer3 = nn.Linear(hidden_size /5 +  hidden_size ,output_size)

    def forward(self, head,body , otherFeat):

        out1 = torch.cat([head,body],dim=1).float()
        # print "INPUT->",inp.size()
        out1 = out1.reshape(out1.size()[0],-1)
        # print "INPUT->",inp.size()
        out1 = self.Layer1(out1)
        out2 = self.LayerO(otherFeat)
        outM = torch.cat([out1,out2],dim=1)
        # print "Combined",out1.size(),out2.size(),outM.size()
        x = self.Layer2(outM)
        x = torch.sigmoid(x)
        x = self.Layer3(x)
        x = torch.sigmoid(x)
        return x


class LSTMNetwork(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers , output_size, bidirectional ,datasetLoaderTrain,datasetLoaderTest):
        super(LSTMNetwork, self).__init__()
        self.trainLoader = datasetLoaderTrain
        self.testLoader = datasetLoaderTest

        self.LayerX1 = nn.Linear((HEAD_WORDS + BODY_WORDS)*50 , hidden_size)
        # self.LayerX1 = nn.Linear(26*2 , hidden_size)
        self.LayerX2 = nn.Dropout(0.3)
        self.LayerX3 = nn.Linear(hidden_size,output_size)


        self.LayerH = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
        self.LayerB = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
        
        
        global ExternalFeatureCount
        self.LayerFinal = nn.Linear(output_size + hidden_size*2 + ExternalFeatureCount, output_size)
        # self.Layer3 = nn.Dropout(self.dropout_p)
        
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, head,body , otherFeat):

        inpX = torch.cat([head,body],dim=1)
        inpX = inpX.reshape(inpX.size()[0],-1)
        inpX = self.LayerX1(inpX)
        inpX = self.LayerX2(inpX)
        inpX = self.LayerX3(inpX)
        

        inpH,_ = self.LayerH(head.transpose(0,1))
        
        inpB,(hidden,_) = self.LayerB(body.transpose(0,1))

        inp = torch.cat([inpH[-1],inpB[-1],inpX,otherFeat],dim=1)
        
        output = self.LayerFinal(inp)
        output = nn.functional.log_softmax(output,dim=1)
        
        return output


class BidirectionalEncoder(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers , output_size, bidirectional ,datasetLoaderTrain,datasetLoaderTest):
        super(BidirectionalEncoder, self).__init__()
        self.trainLoader = datasetLoaderTrain
        self.testLoader = datasetLoaderTest

        
        
        self.LayerH = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
        self.LayerB = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
        
        self.WLearnable = nn.Parameter(torch.ones(((2 if bidirectional else 1)*num_layers*hidden_size,output_size),requires_grad=True))
        
        global ExternalFeatureCount
        print "Linear Layer Size ->",output_size+ExternalFeatureCount,ExternalFeatureCount
        self.Layer1 = nn.Linear(output_size+ExternalFeatureCount,output_size)
    
    def forward(self, head,body ,otherFeat):
        
        # print "Size->",head.size(),body.size()
        
        xHead,(hHead,cHead) = self.LayerH(head.transpose(0,1))
        # print "Here1",xHead.size(),hHead.size()
        xBody,(hBody,cBody) = self.LayerB(body.transpose(0,1),(hHead,cHead))
        # print "Here2",xBody.size(),hBody.size()
        x = hBody.transpose(0,1)
        # print "HereX1,",x.size()
        x = x.reshape(x.size()[0],-1,1)
        # print "HereX2",x.size(),self.WLearnable.size()
        W = self.WLearnable.unsqueeze(2).expand(-1,-1,x.size()[0]).transpose(0,2)
        # print "W->",W.size()
        x = torch.bmm(W,x)
        x = x.reshape(x.size()[0],-1)
        # print "Here3",x.size()
        x = nn.functional.softmax(x,dim=1)

        x = self.Layer1(torch.cat([x,otherFeat],dim=1))
        
        # print x
        return x


class RNNwithAttention(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers , output_size, bidirectional ,datasetLoaderTrain,datasetLoaderTest):
        super(RNNwithAttention, self).__init__()
        self.trainLoader = datasetLoaderTrain
        self.testLoader = datasetLoaderTest

        self.LayerX1 = nn.Linear((HEAD_WORDS + BODY_WORDS)*50 , hidden_size)
        self.LayerX2 = nn.Dropout(0.3)
        self.LayerX3 = nn.Linear(hidden_size,output_size)


        self.LayerH = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
        self.LayerB = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
        
        
        global ExternalFeatureCount
        self.LayerFinal1 = nn.Linear(output_size + hidden_size*4 , hidden_size/5)
        self.LayerFinal2 = nn.Linear(hidden_size/5 + ExternalFeatureCount, output_size)
        # self.Layer3 = nn.Dropout(self.dropout_p)
        
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, head,body ,otherFeat):

        inpX = torch.cat([head,body],dim=1)
        inpX = inpX.reshape(inpX.size()[0],-1)
        inpX = self.LayerX1(inpX)
        inpX = self.LayerX2(inpX)
        inpX = self.LayerX3(inpX)
        

        inpH,_ = self.LayerH(head.transpose(0,1))
        
        inpB,(hidden,_) = self.LayerB(body.transpose(0,1))
        # print "DDD",body.transpose(0,1).size(),type(hidden[0])

        # print "InpH",inpH[-1].size(),inpB[-1].size(),inpX.size()
        # print "InpF",inp.size()
        

        
        # print "Hidden:",hidden.size()

        
        ht = hidden[-1] # (batch x hidden_size)
        # print "ht",ht.size()
        hiddenTrans = hidden.transpose(0,1)   # hidden = batch x layers x hidden_size
        # print "hidden Transpoe",hiddenTrans.size()

        attn_weights = torch.bmm(hiddenTrans,torch.stack([ht],dim=2))   #  ( batch x layers x 1)
        # print "attn weihhts 1",attn_weights.size()
        attn_weights = torch.nn.functional.softmax(attn_weights,dim=1) # (batch  x layers x 1)
        # print "MLTIPLU",hiddenTrans.size(),attn_weights.size()
        attn_applied = hiddenTrans * attn_weights  # batch x layers x hidden_size
        # print "attn weihhts applied",attn_applied.size()

        attentionVector = attn_applied.sum(dim=1)  # batch x hidden_size
        attentionVector = hiddenTrans.sum(dim=1)  # batch x hidden_size
        # print "Attention vector",attentionVector.size()
        output = torch.cat((ht,attentionVector,inpB[-1]), 1)
        # print "Out->",inpH[-1].size(),output.size(),inpX.size()

        inp = torch.cat([inpH[-1],output,inpX],dim=1)
        
        # output = torch.nn.functional.relu(output)
        # output = torch.nn.functional.log_softmax(output)
        output = self.LayerFinal1(inp)
        output = torch.cat([output,otherFeat],dim=1)
        output = nn.functional.log_softmax(output,dim=1)
        
        return output

    
        


class RNN_X(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers ,output_size,bidirectional,train,test):
        super(RNN_X, self).__init__()

        self.hidden_size = hidden_size
        self.trainLoader = train
        self.testLoader = test
        self.bidirectional = bidirectional
        self.Layer1H = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, bidirectional = self.bidirectional, nonlinearity = 'tanh')
        self.Layer1B = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, bidirectional = self.bidirectional, nonlinearity = 'tanh')
        # self.Layer1 = nn.Linear(input_size,hidden_size)
        self.Layer2 = nn.ReLU()

        # self.Layer3 = nn.Linear(hidden_size*2*(2 if self.bidirectional else 1) + 50,100)
        self.Layer3 = nn.Linear(250,50)
        self.Layer4 = nn.Dropout(p=0.3)
        self.Layer5 = nn.Linear(50,output_size)
        self.Layer6 = nn.Dropout(p=0.3)
        self.Layer7 = nn.LogSoftmax(dim=0)
    def forward(self, head,body , baseline):
        # print "INPUTER->",type(x)
        
        outH,_ = self.Layer1H(head)
        outB,_ = self.Layer1B(body)
        # print 'bfore->',baseline.size()
        baseline = baseline.transpose(0,1)
        baseline = baseline.reshape(baseline.size()[0],-1)
        # print 'after->',baseline.size(),'\n'
        # print outH.size(),outB.size(),baseline.size(),outH[-1].size(),baseline.reshape(baseline.size()[1],-1).size()       
        # x = torch.cat([outH[-1],outB[-1],baseline.reshape(baseline.size()[1],-1)],1)
        # print x.size()
        x = self.Layer2(baseline)
        x = self.Layer3(x)
        x = self.Layer4(x)
        x = self.Layer5(x)
        x = self.Layer6(x)
        x = self.Layer7(x)
        return x

class RNN_Y(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers ,output_size,bidirectional,train,test):
        super(RNN_Y, self).__init__()
        self.trainLoader = train
        self.testLoader = test
        
        
        self.Layer1 = nn.Linear(input_size,output_size) 
        # self.Layer2 = torch.nn.Dropout(p=0.7)
        # self.Layer3 = nn.Linear(600,600) 
        # self.Layer4 = torch.nn.Dropout(p=0.7)
        # self.Layer5 = nn.Linear(600,4)    
        # self.Layer6 = nn.LogSoftmax(dim=0)

    def forward(self, x):
        # print "Forward->",x.size()
        x = self.Layer1(x)
        # x = self.Layer2(x)
        # x = self.Layer3(x)
        # x = self.Layer4(x)
        # x = self.Layer5(x)
        # x = self.Layer6(x)
        return x



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


def train(network,epochs,learningRate,classWeight,FNCRun=False):
    global BODY_WORDS
    print "Network Training Starting Now!"
    
    optimizer = optim.Adam(network.parameters(), lr = learningRate)
    trainingLoss = []
    file = open("LogBAG.txt",'w')
    file.write('Epoch,Loss,F1,Accuracy,FNC Score,Train F1,Train Acc,Train FNC Score\n')
    predictionBestMeasure = 0
    predictionBest = None

    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        miniBatch = 0
        for head,body,otherFeat,label in network.trainLoader:
            # print "Train Function Size:",head.size(),body.size(),label.size()
            # sys.stdout.flush()
            
            optimizer.zero_grad()
            label = label.long()
            miniBatch += 1
            
            out = network(head,body,otherFeat)
            # print "ORiginal label:",label.argmax(dim=1)
            # print "Predicted:",out.argmax(dim=1)
            
            origLabel = label.argmax(dim=1)
            loss = criterion(out, origLabel )
            loss.backward()
            optimizer.step()
        
        temp = test(network,network.testLoader,FNCRun),test(network,network.trainLoader,FNCRun)
        if temp[0][1] > predictionBestMeasure:
            # print temp[0][1]
            # print temp[0][3]
            predictionBestMeasure = temp[0][1]
            predictionBest = temp[0][3]
            xxx = open("Predictions.csv",'w')
            xxx.write("TrueLabel,Prediction")
            for x,y in zip(predictionBest[0].data,predictionBest[1].data):
                xxx.write("\n"+str(x.item())+","+str(y.item()))
            xxx.close()
        # inf = ("Epoch: %.3d,  Loss: %.4f \n\t %5s %7s %7s \n\t %5s %.5f %.5f \n\t %5s %.5f %.5f \n\t %5s %.5f %.5f ") % (epoch,loss.item(),"","Test","Train","F1:",temp[0][0],temp[1][0],"Acc:",temp[0][1],temp[1][1],"FNC:",temp[0][2],temp[1][2])
        inf = ("Epoch: %.3d,  Loss: %.5f,  F1: %.3f,  Acc: %.3f,  FNC: %.3f    (%.3f,%.3f,%.3f) ") % (epoch,loss.item(),temp[0][0],temp[0][1],temp[0][2],temp[1][0],temp[1][1],temp[1][2])
        print inf
        file.write("\n%d,%f,%f,%f,%f,%f,%f,%f" % (epoch,loss.item(),temp[0][0],temp[0][1],temp[0][2],temp[1][0],temp[1][1],temp[1][2]))
        file.flush()
    file.close()


def test(network , loader , FNCRun = False):
    predLabel = torch.tensor([],dtype=torch.long)
    origLabel = torch.tensor([],dtype=torch.long)
    for head,body,otherFeat,label in loader:
        # print "Test->",data.size()
        out = network(head,body,otherFeat)
        
        predLabel = torch.cat([predLabel,out.argmax(dim=1)])
        origLabel = torch.cat([origLabel,label.argmax(dim=1)])
        
        # print out
        # print out.argmax(dim=1)
    
    # print("Length of train set:",len(predLabel))
    F1 = round(f1_score(predLabel,origLabel,average='macro'),3)
    Acc = round(accuracy_score(predLabel,origLabel),3)
    
    if FNCRun:
        return F1,Acc,calculateFNC(origLabel,predLabel),(origLabel,predLabel)
    
    
    return F1,Acc,-1.0,(origLabel,predLabel)



def runNetwork(datasetLoaderTrain,datasetLoaderTest,inputSize,hiddenSize,numLayers,outputSize,bidirectional,epochs,lr,classWeight):
    sample = None
    # rnn = RNNwithAttention(inputSize, hiddenSize, numLayers , outputSize, bidirectional ,datasetLoaderTrain,datasetLoaderTest)
    rnn = LSTMNetwork(inputSize, hiddenSize, numLayers , outputSize, bidirectional ,datasetLoaderTrain,datasetLoaderTest)
    # rnn = SimpleNN(inputSize, hiddenSize, numLayers , outputSize, bidirectional ,datasetLoaderTrain,datasetLoaderTest)
    # rnn = BidirectionalEncoder(inputSize, hiddenSize, numLayers , outputSize, True ,datasetLoaderTrain,datasetLoaderTest)    
    
    # count = 0
    # print "_______"
    # for z in rnn.parameters():
    #     count += 1
    #     print "Z",count,z.size()
        
    # print "_______"
    # sys.exit()
    rnn.train()
    train(rnn,epochs,lr,classWeight,True)
    


def runForKaggle():
    global word2VecObjX

    df = readData(DATA_PATH)#[:100]
    # df = shuffle(df)
    ind = int(0.8*len(df))
    dfTrain = df[:ind]
    dfTest = df[ind:]
    kaggleDataTrain = FixedWidth(dfTrain,word2VecObjX,{})
    kaggleDataTest = FixedWidth(dfTest,word2VecObjX,{})
    datasetLoaderTrain = torch.utils.data.DataLoader(kaggleDataTrain, batch_size=100, shuffle=False, num_workers=2)
    datasetLoaderTest = torch.utils.data.DataLoader(kaggleDataTest, batch_size=100, shuffle=False, num_workers=2)
    return datasetLoaderTrain,datasetLoaderTest,kaggleDataTrain.weight

def runForFNC():
    global word2VecObjX
    dfTrain = readFNCData(DATA_PATH_FNC,'train',0)[:500]
    dfTest = readFNCData(DATA_PATH_FNC,'competition_test',0)[:100]
    tfB = None
    dic = getMostCommonWordsDic(dfTrain)
    FNCDataTrain = FixedWidth(dfTrain,word2VecObjX,dic,'file01')
    FNCDataTest = FixedWidth(dfTest,word2VecObjX,dic,'file02')
    
    datasetLoaderTrain = torch.utils.data.DataLoader(FNCDataTrain, batch_size=100, shuffle=False, num_workers=4)
    datasetLoaderTest = torch.utils.data.DataLoader(FNCDataTest, batch_size=100, shuffle=False, num_workers=4)
    return datasetLoaderTrain,datasetLoaderTest,FNCDataTrain.weight  

def runForFNCProblem1():
    global word2VecObjX
    dfTrain = readFNCData(DATA_PATH_FNC,'train',1)#[:400]
    dfTest = readFNCData(DATA_PATH_FNC,'competition_test',1)#[:100]
    tfB = None
    dic = {}#getMostCommonWordsDic(dfTrain)
    FNCDataTrain = FixedWidth(dfTrain,word2VecObjX,dic,'file11')
    FNCDataTest = FixedWidth(dfTest,word2VecObjX,dic,'file12')
    
    datasetLoaderTrain = torch.utils.data.DataLoader(FNCDataTrain, batch_size=500, shuffle=False, num_workers=0)
    datasetLoaderTest = torch.utils.data.DataLoader(FNCDataTest, batch_size=500, shuffle=False, num_workers=0)
    return datasetLoaderTrain,datasetLoaderTest,FNCDataTrain.weight    

def runForFNCProblem2():
    global word2VecObjX
    dfTrain = readFNCData(DATA_PATH_FNC,'train',2)#[:500]
    dfTest = readFNCData(DATA_PATH_FNC,'competition_test',2)#[:100]
    tfB = None
    dic = {}#getMostCommonWordsDic(dfTrain)
    FNCDataTrain = FixedWidth(dfTrain,word2VecObjX,dic,'file21')
    FNCDataTest = FixedWidth(dfTest,word2VecObjX,dic,'file22')
    
    datasetLoaderTrain = torch.utils.data.DataLoader(FNCDataTrain, batch_size=250, shuffle=False, num_workers=4)
    datasetLoaderTest = torch.utils.data.DataLoader(FNCDataTest, batch_size=250, shuffle=False, num_workers=4)
    return datasetLoaderTrain,datasetLoaderTest,FNCDataTrain.weight    


   
def execute():
    global HEAD_WORDS
    global BODY_WORDS

    # datasetLoaderTrain,datasetLoaderTest,classWeight = runForFNC(); outputSize = 4
    datasetLoaderTrain,datasetLoaderTest,classWeight = runForFNCProblem1(); outputSize = 2
    # datasetLoaderTrain,datasetLoaderTest,classWeight = runForFNCProblem2(); outputSize = 3

    # datasetLoaderTrain,datasetLoaderTest,classWeight = runForKaggle(); outputSize = 2
    
    inputSize = 50#*(HEAD_WORDS+BODY_WORDS)
    hiddenSize = 10
    numLayers = 10
    epochs = 2000
    bidirectional = False
    learningRate = 0.01
    
    runNetwork(datasetLoaderTrain,datasetLoaderTest,inputSize,hiddenSize,numLayers,outputSize,bidirectional,epochs,learningRate,classWeight)


execute()