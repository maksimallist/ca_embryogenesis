#
# """
# ==================
# Heavy Metal Umlaut
# ==================
# Example using unicode strings as graph labels.
# Also shows creative use of the Heavy Metal Umlaut:
# https://en.wikipedia.org/wiki/Heavy_metal_umlaut
# """
#
# import matplotlib.pyplot as plt
# import networkx as nx
#
# hd = "H" + chr(252) + "sker D" + chr(252)
# mh = "Mot" + chr(246) + "rhead"
# mc = "M" + chr(246) + "tley Cr" + chr(252) + "e"
# st = "Sp" + chr(305) + "n" + chr(776) + "al Tap"
# q = "Queensr" + chr(255) + "che"
# boc = "Blue " + chr(214) + "yster Cult"
# dt = "Deatht" + chr(246) + "ngue"
#
# G = nx.Graph()
# G.add_edge(hd, mh)
# G.add_edge(mc, st)
# G.add_edge(boc, mc)
# G.add_edge(boc, dt)
# G.add_edge(st, dt)
# G.add_edge(q, st)
# G.add_edge(dt, mh)
# G.add_edge(st, mh)
#
# # write in UTF-8 encoding
# fh = open("edgelist.utf-8", "wb")
# nx.write_multiline_adjlist(G, fh, delimiter="\t", encoding="utf-8")
#
# # read and store in UTF-8
# fh = open("edgelist.utf-8", "rb")
# H = nx.read_multiline_adjlist(fh, delimiter="\t", encoding="utf-8")
#
# for n in G.nodes():
#     if n not in H:
#         print(False)
#
# print(list(G.nodes()))
#
# pos = nx.spring_layout(G)
# nx.draw(G, pos, font_size=16, with_labels=False)
# for p in pos:  # raise text positions
#     pos[p][1] += 0.07
# nx.draw_networkx_labels(G, pos)
# plt.show()

#######################################################################################################################

"""
====================
Parallel Betweenness
====================
Example of parallel implementation of betweenness centrality using the
multiprocessing module from Python Standard Library.
The function betweenness centrality accepts a bunch of nodes and computes
the contribution of those nodes to the betweenness centrality of the whole
network. Here we divide the network in chunks of nodes and we compute their
contribution to the betweenness centrality of the whole network.
"""

from multiprocessing import Pool
import time
import itertools

import matplotlib.pyplot as plt
import networkx as nx


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_source,
        zip([G] * num_chunks, [True] * num_chunks, [None] * num_chunks, node_chunks),
    )

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c


G_ba = nx.barabasi_albert_graph(1000, 3)
G_er = nx.gnp_random_graph(1000, 0.01)
G_ws = nx.connected_watts_strogatz_graph(1000, 4, 0.1)
for G in [G_ba, G_er, G_ws]:
    print("")
    print("Computing betweenness centrality for:")
    print(nx.info(G))
    print("\tParallel version")
    start = time.time()
    bt = betweenness_centrality_parallel(G)
    print(f"\t\tTime: {(time.time() - start):.4F} seconds")
    print(f"\t\tBetweenness centrality for node 0: {bt[0]:.5f}")
    print("\tNon-Parallel version")
    start = time.time()
    bt = nx.betweenness_centrality(G)
    print(f"\t\tTime: {(time.time() - start):.4F} seconds")
    print(f"\t\tBetweenness centrality for node 0: {bt[0]:.5f}")
print("")

nx.draw(G_ba, node_size=100)
plt.show()
