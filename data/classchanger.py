import glob
import os

label_directory = './roaddata/valimg/reallabels'
fu = glob.glob(label_directory + '/*.txt')
#if not os.path.exists('hello'):
#   os.mkdir('hello')
for arxiv, i in enumerate(fu) :
    filename = i.split('/')
    targetname = filename[4]
    f = open(i,'r')

    targetfiledirectory = './roaddata/valimg/labels/'+targetname
    target = open(targetfiledirectory,'w+')
    for x in f:
        a = x.split(' ')
        a[0] = '0'
        a = ' '.join(a)
        target.write(a)
    target.close()
    f.close()