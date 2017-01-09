import os
import csv
import random
import glob

ROOT = '/mnt/hdd/lytton/dataset/VehicleID'
output = '/mnt/hdd/lytton/mx_data/VehicleID'


def load_info(mode):
    infos = file('%s/attribute/%s_attr.txt' % (ROOT, mode)).read().splitlines()
    data = {}
    for x in infos:
        x = x.strip().split(' ')
        data[x[0]] = int(x[1])
    
    return data

###
#generate 3 lists
#train.lst: original images (shuffle or not)
#train-even.lst: even shuffle images
#train-rand.lst: same images with train-even.lst but all shuffled
###

def gen(mode, fin, models,  ifshuffle, split=None, modelmap=None):
    img_lst, cnt = [], 0
    images = [x.strip() for x in file('%s/train_test_split/%s.txt' % (ROOT, fin)).read().splitlines()]

    print len(images)
    if modelmap is None:
        modelmap = {}
        needmap = True
    else:
        needmap = False
    m = 0
    for x in images:
        x = x.strip().split(' ')
        image, vid = x[0], x[1]
        image = '%s/image/%s.jpg' % (ROOT, image)
        if needmap:
            if vid not in modelmap:
                modelmap[vid] = m
                m += 1
        elif vid not in modelmap:
            print "error modelmap"
 #       if vid not in models:
#            print x, vid
        img_lst.append( (cnt, modelmap[vid], models[vid] if vid in models else 250, image) )
        cnt += 1
    print mode, m
    return

    if split is None:
        save(img_lst, mode, ifshuffle)
    else:
        random.shuffle(img_lst)
        k = int(len(img_lst) * split)
        save(img_lst[:k], mode + '-split1')
        save(img_lst[k:], mode + '-split2')
    
    return modelmap


def save(img_lst, fout, ifshuffle=True):
    fo = csv.writer(open('%s/%s.lst' % (output, fout), "w"), delimiter='\t', lineterminator='\n')
    if ifshuffle: random.shuffle( img_lst )
    for item in img_lst:
        fo.writerow( item )

    fo = csv.writer(open('%s/%s-even.lst' % (output, fout), "w"), delimiter='\t', lineterminator='\n')
    img_lst = sorted(img_lst, key=lambda x: x[1])
    lst = []
    model, now, p = -1, 0, -1
    cnt = 0
    for item in img_lst:
        if item[1] != model:
            if now == 1:
                print "single", item, model
            if now % 2 == 1:
                lst.append((cnt, lst[p][1], lst[p][2], lst[p][3]))
                cnt += 1
            p = len(lst)
            model = item[1]
            now = 0
        lst.append((cnt, item[1], item[2], item[3]))
        cnt += 1
        now += 1
    for item in lst:
        fo.writerow( item ) 


    fo = csv.writer(open('%s/%s-rand.lst' % (output, fout), "w"), delimiter='\t', lineterminator='\n')
    random.shuffle( lst )
    for item in lst:
        fo.writerow( item )

if __name__ == '__main__':
    models = load_info('model')
    gen('train', 'train_list', models, ifshuffle=True)
    gen('test-800', 'test_list_800', models, ifshuffle=False)
    # gen('query', 'query', ifshuffle=False, modelmap=modelmap)
    #gen('train', 'bounding_box_train', ifshuffle=True, split=1.7)
    
