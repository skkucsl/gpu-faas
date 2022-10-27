import igraph
import os

os.popen("sleep 1")
igraph.Graph.Barabasi(10000, 10)
os.popen("sleep 2")
igraph.Graph.Barabasi(10000, 10)
os.popen("sleep 3")
