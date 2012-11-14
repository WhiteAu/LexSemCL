from wsd import *

(train_acc, test_acc, test_pred) = runExperimentParsing(
    'Science.tr', 
    'Science.te', 
    complexFFeatures, 
    simpleEFeatures, 
    complexPairFeatures, 
    quietVW=True,
    trainingPOSFile='Science-parsed.tr.pos',
    testPOSFile='Science-parsed.de.pos')
print 'training accuracy =', train_acc
print 'testing  accuracy =', test_acc
h = open('wsd_output', 'w')
for x in test_pred:
    h.write(str(x[0]))
    h.write('\n')
h.close()
