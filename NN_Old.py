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

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

DATA_PATH = 'fake-news-detection/data.csv'
DATA_PATH_FNC = 'fnc-1-master/' 
WORD2VEC_FILEPATH = 'glove.6B.50d.txt'
HEAD_WORDS = 5
BODY_WORDS = 8


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
    df = pd.read_csv(PATH,encoding="utf-8")[:500] 
    df['Headline'] = df['Headline'].apply(lambda val: unicodedata.normalize('NFKD', unicode(val)).encode('ascii', 'ignore').decode())
    df['Body'] = df['Body'].apply(lambda val: unicodedata.normalize('NFKD', unicode(val)).encode('ascii', 'ignore').decode())
    return df

def readFNCData(PATH,fileType):
    dfStances = pd.read_csv(PATH+fileType+'_stances.csv',encoding="utf-8")
    dfBodies = pd.read_csv(PATH+fileType+'_bodies.csv',encoding="utf-8")
    bodyDic = {}
    headDic = {}
    for ind in range(len(dfBodies)):
        bodyDic[dfBodies['Body ID'].iloc[ind]] = dfBodies['articleBody'].iloc[ind]
    
    # print bodyDic.keys()
    return dfStances,bodyDic
    
def padList(lst,l,padder):
    while(len(lst)<l):
        lst += [padder]
    return lst

class FakeNewsKaggle(Dataset):
    def __init__(self,df,word2VecObj):
        
       
        
        self.df = df
        self.w2v = word2VecObj
        self.res = {}
        global HEAD_WORDS
        global BODY_WORDS

        for idx in range(len(self.df)):
            label = torch.zeros(2,dtype=torch.float)
            label[df['Label'].iloc[idx]] = 1.0
            head =  nltk.word_tokenize(df['Headline'].iloc[idx])[:HEAD_WORDS]
            body =  nltk.word_tokenize(df['Body'].iloc[idx])[:BODY_WORDS]

            head = padList(head,HEAD_WORDS,'!!___!!')            
            body = padList(body,BODY_WORDS,'!!___!!')

            head = map(lambda x: torch.tensor(self.w2v.getVec(x)),head)
            body = map(lambda x: torch.tensor(self.w2v.getVec(x)),body)
            

            # print len(head),type(head)
            self.res[idx] = torch.stack(body+head),label

        
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print "Fetching sample",idx," length:",len(self.res[idx][0])," Tensor Size:",self.res[idx][0][0].size()
        inp,label = self.res[idx]
        # print "TENSOR GET ITEM->",inp.size()
        # print inp.size()
        # print label.size()
        # inp = map(lambda x: torch.tensor(self.w2v.getVec(x),dtype=torch.float),inp)
        return inp,label
        
class FNC(Dataset):
    def __init__(self,dfStance,bodyDic,word2vec):
        global HEAD_WORDS
        global BODY_WORDS

        self.dfStance = dfStance
        self.bodyDic = bodyDic
        self.w2v = word2vec
        self.res = {} 
        self.dfStance['Body'] = self.dfStance['Body ID'].apply(lambda x: self.bodyDic[x])
        df = self.dfStance
        
        self.label = {'unrelated':torch.tensor([1,0,0,0],dtype=torch.float) , 'agree':torch.tensor([0,1,0,0],dtype=torch.float) , 'disagree':torch.tensor([0,0,1,0],dtype=torch.float) , 'discuss':torch.tensor([0,0,0,1],dtype=torch.float)}
        df['Label'] =  self.dfStance['Stance'].apply(lambda val: self.label[val])
        
        temp = df.groupby('Stance').count()['Headline'],df.groupby('Stance').count()['Headline']
        agree =  temp[0][0]
        disagree = temp[0][1]
        discuss = temp[0][2]
        unrelated = temp[0][3]
        self.weight =  torch.tensor([1.0/unrelated , 1.0/agree, 1.0/disagree, 1.0/discuss])

        for idx in range(len(df)):
            if (idx+1)%100==0:
                print "Processed %d/%d" % (idx+1,len(df))
            
            baselineFeature = torch.tensor(getMyFeatures(df['Headline'].iloc[idx],df['Body'].iloc[idx]) + [0]*16).float()
            # print "My FEATURE SIZE:",baselineFeature.size()
            baselineFeature = baselineFeature.reshape(5,50)
            # head =  nltk.word_tokenize(df['Headline'].iloc[idx])[:HEAD_WORDS]
            # body =  nltk.word_tokenize(df['Body'].iloc[idx])[:BODY_WORDS]
            
            head =  nltk.FreqDist(nltk.word_tokenize(df['Headline'].iloc[idx])).most_common(HEAD_WORDS)
            body =  nltk.FreqDist(nltk.word_tokenize(df['Body'].iloc[idx])).most_common(BODY_WORDS)
            head =  map(lambda x: x[0],head)
            body =  map(lambda x: x[1],body)

            head = padList(head,HEAD_WORDS,'!!___!!')            
            body = padList(body,BODY_WORDS,'!!___!!')



            head = map(lambda x: torch.tensor(self.w2v.getVec(x)),head)
            body = map(lambda x: torch.tensor(self.w2v.getVec(x)),body)
            
            t1 = torch.stack(body+head)
            # print "DATA LOADER-"
            # print t1.size(),baselineFeature.size()
            t2 = torch.cat([baselineFeature,t1],dim=0)
            # print t2.size()

            self.res[idx] = t2,df['Label'].iloc[idx]

    def __len__(self):
        return len(self.dfStance)
    
    def __getitem__(self, idx):
        inp,label = self.res[idx]
        # print inp.size()
        # print label.size()
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

        self.hidden_size = hidden_size
        self.trainLoader = train
        self.testLoader = test
        self.bidirectional = bidirectional
        self.input_size = input_size
        global HEAD_WORDS
        global BODY_WORDS
        self.Layer = []
        
        self.Layer1 = nn.Linear(input_size,600) 
        self.Layer2 = torch.nn.Dropout(p=0.3)
        self.Layer3 = nn.Linear(600,600) 
        self.Layer4 = torch.nn.Dropout(p=0.3)
        self.Layer5 = nn.Linear(600,4)    
        self.Layer6 = nn.LogSoftmax(dim=0)

    def forward(self, x):
        # print "INPUTER->",type(x)
        # print head.size(),body.size()
        x = x.float()
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
        x = self.Layer5(x)
        x = self.Layer6(x)
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
    
    optimizer = optim.Adagrad(network.parameters(), lr = learningRate)
    trainingLoss = []
    file = open("LogBAG.txt",'w')
    file.write('Epoch,Loss,F1,Accuracy,FNC Score,Train F1,Train Acc,Train FNC Score\n')
    
    sampleLabel = None
    for d,l in network.trainLoader:
        sampleLabel = l
        break

    criterion = nn.MSELoss()
    for epoch in range(1, epochs+1):
        miniBatch = 0
        for data,label in network.trainLoader:
            data = data.transpose(0,1)
            dataBaseline = data[:5,:,:]
            dataB = data[5:BODY_WORDS+5,:,:]
            dataH = data[BODY_WORDS+5:,:,:]
            
            # print "BASELINE:",dataBaseline.size()
            # print "dataB:",dataB.size()
            # print "dataH:",dataH.size()
            optimizer.zero_grad()
            label = label.long()
            # data = torch.nn.utils.rnn.pack_sequence(data)
            miniBatch += 1
            
            
            out = network(dataB,dataH,dataBaseline)
            
            origLabel = label.argmax(dim=1)
            # print label
            # print "LABEL R\n",label
            # print "LABEL S\n",origLabel
            
            # loss = criterion(out, origLabel )
            loss = criterion(out, label.float() )
            
            # loss = criterion(output, label.argmax(dim=1))#if loss is None else loss +  criterion(output, label)
            loss.backward()
            optimizer.step()
        
        temp = test(network,network.testLoader,FNCRun),test(network,network.trainLoader,FNCRun)
        
        # inf = ("Epoch: %.3d,  Loss: %.4f \n\t %5s %7s %7s \n\t %5s %.5f %.5f \n\t %5s %.5f %.5f \n\t %5s %.5f %.5f ") % (epoch,loss.item(),"","Test","Train","F1:",temp[0][0],temp[1][0],"Acc:",temp[0][1],temp[1][1],"FNC:",temp[0][2],temp[1][2])
        inf = ("Epoch: %.3d,  Loss: %.5f,  F1: %.3f,  Acc: %.3f,  FNC: %.3f    (%.3f,%.3f,%.3f) ") % (epoch,loss.item(),temp[0][0],temp[0][1],temp[0][2],temp[1][0],temp[1][1],temp[1][2])
        print inf
        file.write("\n%d,%f,%f,%f,%f,%f,%f,%f" % (epoch,loss.item(),temp[0][0],temp[0][1],temp[0][2],temp[1][0],temp[1][1],temp[1][2]))
        file.flush()
    file.close()

def test(network , loader , FNCRun = False):
    predLabel = torch.tensor([],dtype=torch.long)
    origLabel = torch.tensor([],dtype=torch.long)
    for data,label in loader:
        label = label.long()
        # data = torch.nn.utils.rnn.pack_sequence(data)
        data = data.transpose(0,1)
        dataBaseline = data[:5,:,:]
        dataB = data[5:BODY_WORDS+5,:,:]
        dataH = data[BODY_WORDS+5:,:,:]
        out = network(dataB,dataH,dataBaseline)
        # print output,label
        # print predLabel.size(),(torch.topk(output,k=1)[1]),(torch.topk(output,k=1)[1]).size()
        
        # print out.argmax(dim=1)
        # print label.argmax(dim=1)
        predLabel = torch.cat([predLabel,out.argmax(dim=1)])
        origLabel = torch.cat([origLabel,label.argmax(dim=1)])
        
        # print "OUTPUT->",predLabel.size(),origLabel.size()
        # print "\nPred Label\n",predLabel
        
        # print "\n\nLabel\n",label
        # print "\nOrig Label\n",origLabel
        # print getClass(output),getClass(label[0]),"\n"
    
    F1 = round(f1_score(predLabel,origLabel,average='macro'),3)
    Acc = round(accuracy_score(predLabel,origLabel),3)
    
    if FNCRun:
        return F1,Acc,calculateFNC(origLabel,predLabel)
    
    
    return F1,Acc,-1.0



def runNetwork(datasetLoaderTrain,datasetLoaderTest,inputSize,hiddenSize,numLayers,outputSize,bidirectional,epochs,lr,classWeight):
    rnn = RNN_X(inputSize, hiddenSize, numLayers , outputSize, bidirectional ,datasetLoaderTrain,datasetLoaderTest)
    train(rnn,epochs,lr,classWeight,True)
    # birnn = BiRNN(inputSize, hiddenSize, 1 ,  outputSize,datasetLoaderTrain,datasetLoaderTest)
    # birnn.train(epochs,lr)
    # birnn.test()
    


def runForKaggle():
    global word2VecObjX

    df = readData(DATA_PATH)#[:500]
    # df = shuffle(df)
    ind = int(0.8*len(df))
    dfTrain = df[:ind]
    dfTest = df[ind:]
    kaggleDataTrain = FakeNewsKaggle(dfTrain,word2VecObjX)
    kaggleDataTest = FakeNewsKaggle(dfTest,word2VecObjX)
    datasetLoaderTrain = torch.utils.data.DataLoader(kaggleDataTrain, batch_size=200, shuffle=False, num_workers=0)
    datasetLoaderTest = torch.utils.data.DataLoader(kaggleDataTest, batch_size=200, shuffle=False, num_workers=0)
    return datasetLoaderTrain,datasetLoaderTest
    
def runForFNC():
    global word2VecObjX
    dfTrain = readFNCData(DATA_PATH_FNC,'train')
    dfTest = readFNCData(DATA_PATH_FNC,'competition_test')
    # dfTest = readFNCData(DATA_PATH_FNC,'test')
    dfTrainData = dfTrain[0].copy()[:500]#.iloc[:500,:].copy()
    FNCDataTrain = FNC(dfTrainData,dfTrain[1],word2VecObjX)
    FNCDataTest = FNC(dfTest[0].copy()[:200],dfTest[1],word2VecObjX)
    datasetLoaderTrain = torch.utils.data.DataLoader(FNCDataTrain, batch_size=100, shuffle=False, num_workers=1)
    datasetLoaderTest = torch.utils.data.DataLoader(FNCDataTest, batch_size=100, shuffle=False, num_workers=1)
    return datasetLoaderTrain,datasetLoaderTest,FNCDataTrain.weight    
     
   
def execute():
    global HEAD_WORDS
    global BODY_WORDS

    datasetLoaderTrain,datasetLoaderTest,classWeight = runForFNC(); outputSize = 4
    # datasetLoaderTrain,datasetLoaderTest = runForKaggle(); outputSize = 2
    
    inputSize = 50#*(HEAD_WORDS+BODY_WORDS)
    hiddenSize = 16
    numLayers = 5
    epochs = 1000
    bidirectional = False
    learningRate = 0.005
    
    runNetwork(datasetLoaderTrain,datasetLoaderTest,inputSize,hiddenSize,numLayers,outputSize,bidirectional,epochs,learningRate,classWeight)


execute()