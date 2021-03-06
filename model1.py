from util import *

# given a ttable, where ttable[e][f] = probability of french word f
# given english word e, and a sentence E (list of words) in english
# and F (list of words) in french, compute expected alignments between
# these two sentences
def alignSentencePair(ttable, E, F):
    # ttable[e][f] should be p(f|e)
    N  = len(E)
    M  = len(F)
    al = {}
    # al[n][m] = p(alignment to n | from m, e_n, f_m)
    #          = (1/Z) p(alignment to n, and f_m | from m, e_n)                       -- chain rule
    #          = (1/Z) p(alignment to n | from m, e_n) p(f_m | from m, e_n)      -- chain rule
    #          = (1/Z) p(alignment to n) p(f_m | e_n)            -- independence assumption
    #          = (const) p(f_m | e_n)
    #  where const ensures that sum_n al[n][m] = 1 for all m
    Z = {}
    
    
    for n in range(N):
        al[n] = Counter()
    ### TODO: YOUR CODE HERE    
        for m in range(M):
            if m not in Z:
                Z[m] = 0
                Z[m] = Z[m] + 0
            
            al[n][m] = ttable[E[n]][F[m]]  # = p(f_m | e_n)
            #print Z[m]
            Z[m] += ttable[E[n]][F[m]]
            #print 'al[n][m] is: %f'%al[n][m]
    for n in al.iterkeys():
        for m in al[n].iterkeys():
            al[n][m] *= float(1/Z[m])      # I think 1/sum(p(f|e)) is the normalizing factor... 

    return al

# given a ttable (as above), two sentences (as above) and an alignment
# (as *created* above), add fractional counts to the ttable
# corresponding to the aligned words
def addFractionalCounts(ttable, E, F, al):
    for n in al.iterkeys():
        #print 'n is : %s'%n
        for m in al[n].iterkeys():
            #print 'm is : %s'%m
            #print 'E[n] is: %s, F[m] is: %s'%(E[n], F[m])
            #print 'al[n][m] is: %f'%al[n][m]
            if E[n] not in ttable:
                ttable[E[n]] = Counter()
            if F[m] not in ttable[E[n]]:
                ttable[E[n]][F[m]] = 0
            ttable[E[n]][F[m]] += al[n][m]
        #Z = sum( [ttable[E[n]][x] for x in F])
        #for m in al[n].iterkeys():
        #    ttable[E[n]][F[m]] *= float(1/Z) #normalize ttable entries by new prob amount.
        ttable[E[n]].normalize()
        

def singleEMStep(ttable, corpus, printAlignments=False):
    newTTable = {}
    for (E,F) in corpus:
        al = alignSentencePair(ttable, E, F)
        addFractionalCounts(newTTable, E, F, al)
        if printAlignments:
            print "F = ", F
            for n in range(len(E)):
                print "  ", E[n], ": ", str(al[n])
            print "-----------------------------------------------------------------"

    for e in newTTable.iterkeys():
        newTTable[e].normalize()

    return newTTable

def uniformTTableInitialization(corpus):
    ttable = {}
    for (E,F) in corpus:
        for e in E:
            if not ttable.has_key(e):
                ttable[e] = Counter()
            for f in F:
                ttable[e][f] = 1
    for e in ttable.iterkeys():
        ttable[e].normalize()
    return ttable

def runEM(corpus, ttable0=None, numIter=5, printAll=False):
    if ttable0 is None:
        print 'initializing ttable'
        ttable0 = uniformTTableInitialization(corpus)
    ttable = ttable0
    for it in range(numIter):
        print 'iteration ', str(it+1)
        ttable = singleEMStep(ttable, corpus, printAll)
    return ttable

def printTTable(ttable, outputfilename):
    h = open(outputfilename, 'w')
    for e in ttable.iterkeys():
        for f,p in ttable[e].iteritems():
            if p > 1e-6:
                h.write(e)
                h.write("\t")
                h.write(f)
                h.write("\t")
                h.write(str(p))
                h.write("\n")
            

def removeRareWords(corpus, threshold=25):
    ec = Counter()
    fc = Counter()
    for i in range(len(corpus)):
        for w in corpus[i][0]:
            ec[w] += 1
        for w in corpus[i][1]:
            fc[w] += 1
    for i in range(len(corpus)):
        for j in range(len(corpus[i][0])):
            if ec[w] < threshold:
                corpus[i][0][j] = 'UNK'
        for j in range(len(corpus[i][1])):
            if fc[w] < threshold:
                corpus[i][1][j] = 'UNK'
        

def readCorpus(efile, ffile, truncateWordLength=None, rareWordThreshold=25):
    corpus = []
    eh = open(efile, 'r')
    fh = open(ffile, 'r')
    for estr in eh.readlines():
        E = estr.lower().strip().split()
        F = fh.readline().lower().strip().split()
        if truncateWordLength is not None:
            for i in range(len(E)):
                E[i] = E[i][0:truncateWordLength]
            for i in range(len(F)):
                F[i] = F[i][0:truncateWordLength]
        corpus.append( (E,F) )
    eh.close()
    fh.close()
    removeRareWords(corpus, rareWordThreshold)
    return corpus

def readCorpusSingleFile(onefile):
    corpus = []
    h = open(onefile, 'r')
    for estr in eh.readlines():
        E = estr.strip().split()
        F = h.readline().strip().split()
        corpus.append( (E,F) )
    h.close()
    return corpus


simpleTestCorpus = [
    ( ["la", "maison"], ["the", "house"]         ),
    ( ["la", "maison", "bleue"], ["the", "blue", "house"] ),
    ( ["la", "fleur"], ["the", "flower"])
    ]

if __name__ == '__main__':
    ttable = uniformTTableInitialization(simpleTestCorpus)
    alignSentencePair(ttable, simpleTestCorpus[0][0], simpleTestCorpus[0][1])
    ttable = runEM(simpleTestCorpus)
    #ttable
