from tkinter import N
import nltk

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import numpy as np
import math
import itertools
import networkx as nx
import matplotlib.pyplot as plt

import collections

text = '''
This is a sample text that contains the name Alex Smith who is one of the developers of this project.
You can also find the surname Jones here.
'''

#reading csv file containing dreams
# importing csv module
import csv

# csv file name
filename = "dream_corpus.csv"
 
# initializing the titles and rows list
fields = []
dreams = []
 
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    # extracting field names through first row
    fields = next(csvreader)
    # extracting each data row one by one
    for row in csvreader:
        dreams.append(row)
    # get total number of rows
    print("Total no. of rows: %d"%(csvreader.line_num))
 
# printing the field names
#print('Field names are:' + ', '.join(field for field in fields))
 
#create empty list holding names
name_list = list()
dream_name_list_all = list()
#skip these names (because they are created by mistake)
stop_names = list()

# NLP for five random rows print results
print('\nBelow are the first five dreams:\n')
count = 0
for dream in dreams:
    count = count + 1
    if count <6:
        print('\nDream No. %d\n'%(count))
    # parsing each column of a row
    nltk_results = ne_chunk(pos_tag(word_tokenize(dream[0])))
    #create list containing all names in one single dream
    dream_name_list = []
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
            if count <6:
                print ('Type: ', nltk_result.label(), 'Name: ', name)
            #add name to dream and total name lists
            #add name to single dream's name list
            if nltk_result.label() == 'PERSON': #change to plot networks of other types
                if name not in dream_name_list:
                    dream_name_list.append(name)
                if name not in name_list:
                    name_list.append(name)
    #add list of names to big list for all dreams
    dream_name_list_all.append(dream_name_list)
    #for col in row:
    #    print("%10s"%col,end="\n"),


#check name lists
#print('\nThis is the list of all names that ever appeared:\n')
#print(name_list)

#print('\nThis is the list of each dream name list\n')
#print(dream_name_list_all)

#create matrix for name co-occurance
#create matrix for all name pairs
graph = np.zeros((len(name_list), len(name_list)))
#for each dream's name list, code each pair
for dream_name_list in dream_name_list_all:
    for name_pair in itertools.combinations(dream_name_list, 2):
        graph[name_list.index(name_pair[0])][name_list.index(name_pair[1])] += 1

#view adjacency matrix for network
#print('\nThis is the updated adjacency matrix for the co-occurence network\n')
#print(graph)

#create and plot graph using neworkX
nxgraph = nx.from_numpy_matrix(graph)


#rename nodes with character names
mapping = dict(zip(nxgraph, name_list))
nxgraph = nx.relabel_nodes(nxgraph, mapping)
d = dict(nxgraph.degree)

#plot the distribution of degrees

#set color gradient
def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]
#define color gradient for plot
color1 = "#23abeb"
color2 = "#a2dffc"

#sort dictionary
d_sorted = sorted(d.items(), key=lambda kv: kv[1], reverse = True)
d_sorted = collections.OrderedDict(d_sorted)
keys = d_sorted.keys()
values = d_sorted.values()
plt.bar(list(keys)[0:14], list(values)[0:14], color = get_color_gradient(color1, color2, 15))
plt.xticks(rotation = "vertical", size = 7)
plt.title("Degree centrality of first 15 characters in dreams")
plt.show()

#calculate and print network metrics:
density = nx.density(nxgraph)
transitivity = nx.transitivity(nxgraph)
assortativity = nx.degree_assortativity_coefficient(nxgraph)
print('\nMy social network is characterized by the following metrics:')
print('density: ', "{:.3f}".format(density))
print('transitivity: ', "{:.3f}".format(transitivity))
print('degree assortativity', "{:.3f}".format(assortativity))


#show network plot
#set node size by degree centrality
nx.draw(nxgraph, with_labels=True, 
    node_size=[math.sqrt(v+1) * 500 for v in d.values()])
#set node color by centrality
plt.show()
