import re
import os
from math import *
from util import *
from tree import *

#USE YOUR LOCAL PATH. THIS IS MINE. USE YOURS
vw = os.getenv('VW_PATH','/home/hal/Documents/CompLing/vowpal_wabbit/vowpalwabbit/vw')
#vw = '/home/hal/Documents/CompLing/vowpal_wabbit/vowpalwabbit/vw'

# readWSDCorpus: returns the a LIST of (labelInfo, doc) pairs, where
# labelInfo is a four-tuple (sent #, word #, fWord, eWord).  here,
# fWord is the foreign word string, and eWord is the correct
# translation in this context.  the doc object is a list of lists,
# where doc[i] is the ith sentence (zero indexed, of course) and
# doc[i][j] is the jth word of the ith sentence.  if len(doc[i])==0,
# this means that this is a paragraph break
def readWSDCorpus(filename):
    print 'reading data from ', filename
    matchRE = re.compile("^(.+)___inE=(.+)$")
    data = []
    h = open(filename, 'r')
    doc = []
    lastWasPBreak = True
    for l in h.readlines():
        l = l.strip()
        words = l.split()

        if len(words) == 0:
            if len(doc) > 0 and lastWasPBreak:
                doc.pop()
            if len(doc) > 0:
                labelInfo = []
                for i in range(len(doc)):   # i indexes sentence
                    for j in range(len(doc[i])):   # j indexes words
                        word = doc[i][j]
                        m = re.match(matchRE, word)
                        if m is not None:
                            fWord = m.group(1)
                            eWord = m.group(2)
                            doc[i][j] = fWord
                            labelInfo.append( (i,j,fWord.lower(),eWord.lower()) )

                if len(labelInfo) > 0:
                    data.append( (labelInfo, doc) )

            doc = []
            lastWasPBreak = True

        elif len(words) == 1 and words[0] == "<P>":
            if not lastWasPBreak:
                doc.append([])
            lastWasPBreak = True
        else:
            lastWasPBreak = False
            doc.append(words)

    h.close()
    
    return data

# take a corpus, as return by readWSDCorpus, and construct a ttable of
# possible translations.  ttable[fWord][eWord] is the total count of
# times that we saw fWord translated into eWord
def collectTranslationTable(corpus):
    ttable = {}
    for (labelInfoList,doc) in corpus:
        for labelInfo in labelInfoList:
            fWord = labelInfo[2]
            eWord = labelInfo[3]
            if not ttable.has_key(fWord):
                ttable[fWord] = Counter()
            ttable[fWord][eWord] += 1
    return ttable

# iterate over a corpus (building a ttable if necessary); yields
# 5-tuples of the form (doc, sent #, word #, list of possible
# translations, true translation in that list).  the list of possible
# translations includes pairs (w,p) of the word and its probability.
# the last is the index of the true translation.
def iterateTrainingExamples(corpus, ttable):
    for ( labelInfoList , doc) in corpus:
        for (i,j,fWord,eWord) in labelInfoList:
            if not ttable.has_key(fWord):
                raise Exception('ttable does not contain foreign word "' + fWord + '"')

            possibleTranslations = ttable[fWord].items()
            trueTransId = -1
            for ii in range(len(possibleTranslations)):
                if possibleTranslations[ii][0] == eWord:
                    trueTransId = ii
                    break

            yield (doc, i, j, possibleTranslations, trueTransId)

# take an example info and some feature generating functions and
# construct things to look like a VW-style example
def makeExample(exampleInfo, getFFeatures, getEFeatures, getPairFeatures=None):
    (doc,i,j,possibleTranslations,trueTransId) = exampleInfo
    src = getFFeatures(doc, i, j)

    trans = []
    for k in range(len(possibleTranslations)):
        cost = 0 if k == trueTransId else 1
        tgt  = getEFeatures(possibleTranslations[k][0], possibleTranslations[k][1])
        pair = {} if getPairFeatures is None else getPairFeatures(doc, i, j, possibleTranslations[k][0], possibleTranslations[k][1])

        trans.append( (cost, tgt, pair) )

    return (src, trans)

# write a vw-style example to file
def writeVWExample(h, example, featureSetTracker=None):
    def sanitizeFeature(f):
        return re.sub(':', '_COLON_',
               re.sub('\|', '_PIPE_',
               re.sub('[\s]', '_', f)))
    
    def printFeatureSet(namespace, fdict):
        h.write(' |')
        h.write(namespace)
        for f,v in fdict.iteritems():
            h.write(' ')
            if abs(v) > 1e-6:
                ff = sanitizeFeature(f)
                h.write(ff)
                if abs(v-1) > 1e-6:
                    h.write(':')
                    h.write(str(v))
                if featureSetTracker is None:
                    if not featureSetTracker.has_key(namespace): featureSetTracker[namespace] = {}
                    featureSetTracker[namespace][ff] = f

    (src,trans) = example
    if len(src) > 0:
        h.write('shared')
        printFeatureSet('s', src)
        h.write('\n')
    for i in range(len(trans)):
        (cost, tgt, pair) = trans[i]
        h.write(str(i+1))
        h.write(':')
        h.write(str(cost))
        printFeatureSet('t', tgt)
        printFeatureSet('p', pair)
        h.write('\n')
    h.write('\n')

# take a corpus and actually generate some training data for VW
def generateVWData(corpus, ttable, getFFeatures, getEFeatures, getPairFeatures=None, outputFilename=None):
    completeFeatureSet = {}
    if outputFilename is None:
        outputFilename = 'vw_training_data'
    h = open(outputFilename, 'w')
    for exampleInfo in iterateTrainingExamples(corpus, ttable):
        example = makeExample(exampleInfo, getFFeatures, getEFeatures, getPairFeatures)
        writeVWExample(h, example, completeFeatureSet)
    h.close()
    

def evaluatePredictions(corpus, ttable, predictions):
    exNum = 0
    acc = 0.0
    for (doc, i, j, possibleTranslations, trueTransId) in iterateTrainingExamples(corpus, ttable):
        if exNum >= len(predictions):
            print exNum
            print len(predictions)
            raise Exception('not enough predictions: did you accidentally use the wrong files?')
        (predId, predList) = predictions[exNum]
        if predId == trueTransId:
            acc += 1.0
        exNum += 1

    if exNum < len(predictions):
        print exNum
        print len(predictions)
        raise Exception('too many predictions: did you accidentally use the wrong files?')

    if exNum == 0: return 0.0
    return acc / float(exNum)


def runExperiment(trainingFile, testFile, getFFeatures, getEFeatures, getPairFeatures=None, filePrefix='wsd_vw', quietVW=False):
    trainFileVW = filePrefix + '.tr'
    testFileVW  = filePrefix + '.te'
    modelFileVW = filePrefix + '.model'

    trainingCorpus = readWSDCorpus(trainingFile)
    testCorpus = None if testFile is None else readWSDCorpus(testFile)

    print 'collecting translation table'
    ttable = collectTranslationTable(trainingCorpus)

    print 'generating classification data'
    generateVWData(trainingCorpus, ttable, getFFeatures, getEFeatures, getPairFeatures, trainFileVW)
    if testCorpus is not None:
        generateVWData(testCorpus, ttable, getFFeatures, getEFeatures, getPairFeatures, testFileVW )

    trainVW(trainFileVW, modelFileVW, quietVW)

    train_pred = testVW(trainFileVW, modelFileVW, quietVW)
    train_acc = evaluatePredictions(trainingCorpus, ttable, train_pred)

    test_pred = None
    test_acc = 0
    if testCorpus is not None:
        test_pred = testVW(testFileVW , modelFileVW, quietVW)
        test_acc  = evaluatePredictions(testCorpus, ttable, test_pred)

    return (train_acc, test_acc, test_pred)

# Added this variant on runExperiment per Hal's Piazza suggestion.  -- Alex
def runExperimentParsing(trainingFile, testFile, getFFeatures, getEFeatures, getPairFeatures=None, filePrefix='wsd_vw', quietVW=False, trainingPOSFile=None, testPOSFile=None ):
    trainFileVW = filePrefix + '.tr'
    testFileVW  = filePrefix + '.te'
    modelFileVW = filePrefix + '.model'

    trainingCorpus = readWSDCorpus(trainingFile)
    testCorpus = None if testFile is None else readWSDCorpus(testFile)

    # We need to parse the POS files, both train and test, and pass that to feature creation.
    trainingPOS = readPOSFile(trainingPOSFile)
    testPOS = None if testPOSFile is None else readPOSFile(testPOSFile)

    print 'collecting translation table'
    ttable = collectTranslationTable(trainingCorpus)

    print 'generating classification data'
    generateVWData(trainingCorpus, ttable, getFFeatures, getEFeatures, getPairFeatures, trainFileVW)
    if testCorpus is not None:
        generateVWData(testCorpus, ttable, getFFeatures, getEFeatures, getPairFeatures, testFileVW )

    trainVW(trainFileVW, modelFileVW, quietVW)

    train_pred = testVW(trainFileVW, modelFileVW, quietVW)
    train_acc = evaluatePredictions(trainingCorpus, ttable, train_pred)

    test_pred = None
    test_acc = 0
    if testCorpus is not None:
        test_pred = testVW(testFileVW , modelFileVW, quietVW)
        test_acc  = evaluatePredictions(testCorpus, ttable, test_pred)

    return (train_acc, test_acc, test_pred)

# Get a list[sent][word] of POS out of the file
def readPOSFile(filename):
    #lst = []
    #for tree in iterateTree(filename):
        #lst.append(tree.preterminals())
    #return lst
    lst = []
    fin = open(filename,'r')
    for line in fin:
        lst.append(line.strip().split(','))
    fin.close()
    return lst

# I copied this from P2 extractGrammar.py  -- Alex
def iterateTree(filename):
    h = open(filename, 'r')
    for line in h:
        if line.strip() == '':
            continue
        line = re.sub('\(', '(TOP', line, count=1)
        tree = de_annotate(bracket_parse(line))
        if tree is None: continue

        yield tree
    h.close()


# I copied this from P2 extractGrammar.py; not sure what it does!  -- Alex
def de_annotate(tree):
    if type(tree) is str:
        if len(tree) == 0: return None
        if tree == '-NONE-': return None
        return tree.split('-')[0]

    if len(tree.node) == 0: return None
    if tree.node == '-NONE-': return None
    children = []
    for i,child in enumerate(tree):
        newChild = de_annotate(child)
        if newChild is not None:
            children.append(newChild)
    if len(children) == 0:
        return None
    return Tree(tree.node.split('-')[0], children)

def trainVW(dataFilename, modelFilename, quietVW=False):
    cmd = vw + ' -k -c --passes 10 -q st --power_t 0.5 --csoaa_ldf m -d ' + dataFilename + ' -f ' + modelFilename
    if quietVW: cmd += ' --quiet'
    print 'executing: ', cmd
    p = os.system(cmd)
    if p != 0:
        raise Exception('execution of vw failed!  return value=' + str(p))

def testVW(dataFilename, modelFilename, quietVW=False):
    cmd = vw + ' -t -q st --csoaa_ldf m -d ' + dataFilename + ' -i ' + modelFilename + ' -r ' + dataFilename + '.rawpredictions'
    if quietVW: cmd += ' --quiet'
    print 'executing: ', cmd
    p = os.system(cmd)
    if p != 0:
        raise Exception('execution of vw failed!  return value=' + str(p))
    
    h = open(dataFilename + '.rawpredictions')
    predictions = []

    this = []
    thisBestId  = -1
    thisBestVal = 0
    for l in h.readlines():
        l = l.strip()
        res = l.split(':')
        if len(l) == 0:
            predictions.append( (thisBestId-1, predictions) )
            this = []
            thisBestId = -1
            thisBestVal = 0
        elif len(res) == 2:
            class_id  = int(res[0])
            class_val = float(res[1])
            if thisBestId < 0 or class_val < thisBestVal:
                thisBestId  = class_id
                thisBestVal = class_val
            this.append( (class_id, class_val) )
        else:
            raise Exception('error on vw output, got line "' + l + '"')
    h.close()

    return predictions
