# SS Names from https://www.ssa.gov/oact/babynames/limits.html
from pylab import *
import pandas as pd
import scipy
import os
from itertools import combinations
from sklearn.cluster import AffinityPropagation as AP
from sklearn.cluster import KMeans as KM
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering as SC

floc="data/"
li=os.popen("ls "+floc+"yob*.txt").read().split()
years=np.array([i.strip()[8:-4] for i in li],dtype=int)
freq=years*0


# acquire unique boy and girl names
unique_boy_names=[]
unique_girl_names=[]
for ili in li:
    print(ili)
    data=pd.read_csv(ili,names=["Name","Sex","Frequency"])
    data.Frequency/=data.Frequency.max()
    # data=data[data.Frequency>5e-2]
    data=data[data.Frequency>1e-1]
    unique_boy_names+=data.Name[data.Sex=='M'].values.tolist()
    unique_girl_names+=data.Name[data.Sex=='F'].values.tolist()
unique_boy_names=np.array(unique_boy_names).flatten()
unique_boy_names=np.unique(unique_boy_names)
unique_girl_names=np.array(unique_girl_names).flatten()
unique_girl_names=np.unique(unique_girl_names)


# acquire frequency of these names
bname_frequencies=np.array([years*0 for name in unique_boy_names],dtype=float)
gname_frequencies=np.array([years*0 for name in unique_girl_names],dtype=float)

for year,ili in enumerate(li):
    print(ili,years[year])
    data=pd.read_csv(ili,names=["Name","Sex","Frequency"])
    
    # scale by natural maximum in each year
    data.Frequency/=data.Frequency.max()
    for iname,name in enumerate(data.Name):
        if data.Sex[iname]=='M' and name in unique_boy_names:
            bname_frequencies[unique_boy_names==name,year]=data.Frequency[iname]
        if data.Sex[iname]=='F' and name in unique_girl_names:
            gname_frequencies[unique_girl_names==name,year]=data.Frequency[iname]


clf=AP()
clf.fit(bname_frequencies)

nf=open("boy_names_nodes.json",'w')
nf.write("\"nodes\" : [\n")
ln=len(unique_boy_names)
for i in range(ln):
    name=unique_boy_names[i]
    igroup=str(clf.labels_[i])
    nf.write("\t{\"name\": \""+name+"\",\"group\": "+igroup+"}")
    if i <(len(unique_boy_names)-1):
        nf.write(",\n")
    else:
        nf.write("]\n")
nf.close()

# BEGIN LINKS
lf=open("boy_names_links.json",'w')
lf.write("\"links\": [\n")

centers=clf.cluster_centers_
lc=len(centers)
labels=clf.labels_
ids=[]
ds=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(bname_frequencies))
ds=10*np.exp(-np.abs(ds))

# IDENTIFY CLUSTER CENTERS
for i in range(lc):
    # by closest to center
    # ids.append(np.argsort(np.sum((centers[i]-bname_frequencies)**2,axis=1))[0])
    # by most popular name
    lmax=np.argsort(bname_frequencies[labels==i].mean(axis=1))[-1]
    ids.append(np.where(labels==i)[0][lmax])

# LINK CLUTER CENTERS
for i,j in combinations(ids,2):
    lf.write("\t{"+"\"source\": {0:},\"target\": {1:},\"value\": {2:}".format(i,j,ds[i,j])+"},\n")

for i in range(lc):
    if np.sum(labels==i)==1:
        continue
    clf2=AP()
    clf2.fit(bname_frequencies[labels==i])
    centers2=clf2.cluster_centers_
    lc2=len(centers2)
    labels2=clf2.labels_
    ids2=[]
    # IDENTIFY CLUSTER CENTERS
    for j in range(lc2):
        # by closest to center
        # ids2.append(np.argsort(np.sum((centers2[j]-bname_frequencies)**2,axis=1))[0])
        # by most popular name
        lmax=np.argsort(bname_frequencies[labels==i][labels2==j].mean(axis=1))[-1]
        ids2.append(np.where(labels==i)[0][np.where(labels2==j)[0]][lmax])
    
    # LINK CLUSTER CENTERS TO ORIGINAL CLUSTER
    for j in ids2:
        if j==ids[i]:
            continue
        print("linking {:} and {:}".format(ids[i],j))
        lf.write("\t{"+"\"source\": {0:},\"target\": {1:},\"value\": {2:}".format(ids[i],j,ds[ids[i],j])+"},\n")

    # LINK CLUSTERS TOGETHER
    for k,l in combinations(ids2,2):
        if k==ids[i] or l == ids[0]:
            continue
        print("linking {:} and {:}".format(k,l))
        lf.write("\t{"+"\"source\": {0:},\"target\": {1:},\"value\": {2:}".format(k,l,ds[k,l])+"},\n")

    # LINK SUB CLUSTERS TO SUB CLUSTER CENTERS
    for j in range(lc2):
        k=ids2[j]
        for l in np.where(labels==i)[0][np.where(labels2==j)[0]]:
            if k==l:
                continue
            print("linking {:} and {:}".format(k,l))
            lf.write("\t{"+"\"source\": {0:},\"target\": {1:},\"value\": {2:}".format(k,l,ds[k,l])+"},\n")            
lf.write("]\n")
lf.close()
        


# NOW FOR GIRL NAMES
clf=AP()
clf.fit(gname_frequencies)

nf=open("girl_names_nodes.json",'w')
nf.write("\"nodes\" : [\n")
ln=len(unique_girl_names)
for i in range(ln):
    name=unique_girl_names[i]
    igroup=str(clf.labels_[i])
    nf.write("\t{\"name\": \""+name+"\",\"group\": "+igroup+"}")
    if i <(len(unique_girl_names)-1):
        nf.write(",\n")
    else:
        nf.write("]\n")
nf.close()

# BEGIN LINKS
lf=open("girl_names_links.json",'w')
lf.write("\"links\": [\n")

centers=clf.cluster_centers_
lc=len(centers)
labels=clf.labels_
ids=[]
ds=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(gname_frequencies))
ds=10*np.exp(-np.abs(ds))

# IDENTIFY CLUSTER CENTERS
for i in range(lc):
    # by most popular name
    lmax=np.argsort(gname_frequencies[labels==i].mean(axis=1))[-1]
    ids.append(np.where(labels==i)[0][lmax])

# LINK CLUTER CENTERS
for i,j in combinations(ids,2):
    lf.write("\t{"+"\"source\": {0:},\"target\": {1:},\"value\": {2:}".format(i,j,ds[i,j])+"},\n")

for i in range(lc):
    if np.sum(labels==i)==1:
        continue
    clf2=AP()
    clf2.fit(gname_frequencies[labels==i])
    centers2=clf2.cluster_centers_
    lc2=len(centers2)
    labels2=clf2.labels_
    ids2=[]
    # IDENTIFY CLUSTER CENTERS
    for j in range(lc2):
        # by most popular name
        lmax=np.argsort(gname_frequencies[labels==i][labels2==j].mean(axis=1))[-1]
        ids2.append(np.where(labels==i)[0][np.where(labels2==j)[0]][lmax])
    
    # LINK CLUSTER CENTERS TO ORIGINAL CLUSTER
    for j in ids2:
        if j==ids[i]:
            continue
        print("linking {:} and {:}".format(ids[i],j))
        lf.write("\t{"+"\"source\": {0:},\"target\": {1:},\"value\": {2:}".format(ids[i],j,ds[ids[i],j])+"},\n")

    # LINK CLUSTERS TOGETHER
    for k,l in combinations(ids2,2):
        if k==ids[i] or l == ids[0]:
            continue
        print("linking {:} and {:}".format(k,l))
        lf.write("\t{"+"\"source\": {0:},\"target\": {1:},\"value\": {2:}".format(k,l,ds[k,l])+"},\n")

    # LINK SUB CLUSTERS TO SUB CLUSTER CENTERS
    for j in range(lc2):
        k=ids2[j]
        for l in np.where(labels==i)[0][np.where(labels2==j)[0]]:
            if k==l:
                continue
            print("linking {:} and {:}".format(k,l))
            lf.write("\t{"+"\"source\": {0:},\"target\": {1:},\"value\": {2:}".format(k,l,ds[k,l])+"},\n")            
lf.write("]\n")
lf.close()
        
