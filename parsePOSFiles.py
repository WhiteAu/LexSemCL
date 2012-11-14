from wsddata import *

for fname in ['Science-parsed.tr', 'Science-parsed.te', 'Science-parsed.de']:
    fout = open('%s.pos' % fname,'w')
    for tree in iterateTree(fname):
        fout.write(','.join(tree.preterminals())+'\n')

    fout.close()
