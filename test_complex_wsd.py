from wsd import *

(train_acc, test_acc, test_pred) = runExperiment(
    'Science.tr', 
    'Science.de', 
    simpleFFeatures, 
    simpleEFeatures, 
    complexPairFeatures, 
    quietVW=True)
print 'training accuracy =', train_acc
print 'testing  accuracy =', test_acc
h = open('wsd_output', 'w')
for x in test_pred:
    h.write(str(x[0]))
    h.write('\n')
h.close()
