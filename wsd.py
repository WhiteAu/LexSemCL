import re
import os
from util import *
from wsddata import *

# simpleEFeatures(w) takes a word (in English) and generates relevant
# features.  At the very least, this should include the word identity
# as a feature.  You should also make it include prefix (two
# characters) and suffix (two characters) features.
def simpleEFeatures(w, wprob):
    feats = Counter()     # features are a mapping from strings (names) to floats (values)

    # generate a single feature with the word identity, called
    # w_EWORD, and value 1
    feats['w_' + w] = 1

    # generate a feature called wlogprob whose value is log(wprob)
    feats['wlogprob'] = log(wprob)

    # generate a feature corresponding to the two character prefix of
    # this word and a second feature for the two character suffix.
    # for example, on the word "happiness", generate "pre_ha" and
    # "suf_ss" as features.
    ### TODO: YOUR CODE HERE
    util.raiseNotDefined()

    return feats

# simpleFFeatures(doc, i, j) asks for features about the French word
# in sentence i, position j of doc (i.e., at doc[i][j]).
def simpleFFeatures(doc, i, j):
    w = doc[i][j]

    feats = Counter()

    # generate a single feature with the (french) word identity
    feats[w] = 1

    # generate a feature corresponding to the two character prefix of
    # this word and a second feature for the two character suffix; same
    # deal about pre_ and suf_
    ### TODO: YOUR CODE HERE
    util.raiseNotDefined()

    # generate features corresponding to the OTHER words in this
    # sentence.  for some other word "w", create a feature of the form
    # sc_w, where sc means "sentence context".  if a single word (eg.,
    # "the") appears twice in the context, the feature sc_the should
    # have value two.
    ### TODO: YOUR CODE HERE
    util.raiseNotDefined()

    return feats

# simplePairFeatures(doc, i, j, ew, wprob) -- the first three are the
# same as for simpleFFeatures, the last two are the same as for
# simpleEFeatures.  we return features that are functions of both the
# french word (doc[i][j] and the english word w.
def simplePairFeatures(doc, i, j, ew, wprob):
    fw = doc[i][j]

    feats = Counter()

    # we have just one feature that asks if the fw and ew are
    # identical; this is really only useful for example on proper
    # nouns
    if fw == ew:
        feats['w_eq'] = 1

    return feats

def complexEFeatures(w, wprob):
    feats = Counter()
    return feats

def complexFFeatures(doc, i, j):
    feats = Counter()
    return feats

def complexPairFeatures(doc, i, j, ew, wprob):
    feats = Counter()
    return feats

if __name__ == "__main__":
    (train_acc, test_acc, test_pred) = runExperiment('Science.tr', 'Science.de', simpleFFeatures, simpleEFeatures, simplePairFeatures, quietVW=True)
    print 'training accuracy =', train_acc
    print 'testing  accuracy =', test_acc
    h = open('wsd_output', 'w')
    for x in test_pred:
        h.write(str(x[0]))
        h.write('\n')
    h.close()


    
