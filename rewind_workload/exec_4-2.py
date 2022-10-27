import sys, json
import time
import igraph

def main(args):
    startTime = time.time()

    size = int(args.get('size', '10000'))

    graph_generating_begin = time.time()
    graph = igraph.Graph.Barabasi(size, 10)
    graph_generating_end = time.time()

    process_begin = time.time()
    result = graph.spanning_tree(None, False)
    process_end = time.time()

    graph_generating_time = graph_generating_end - graph_generating_begin
    process_time = process_end - process_begin

    tmp = {'startTime': startTime, 'latency': process_time, 'functionTime': process_end - startTime}
    print(json.dumps(tmp))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        args = dict()
    main(args)
