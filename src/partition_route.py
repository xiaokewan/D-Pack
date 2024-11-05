def parse_route_file(file_path, graph):
    with open(file_path, 'r') as file:
        current_net = None
        for line in file:
            if line.startswith('Net'):
                current_net = line.split()[2].strip('()')
            elif '->' in line:
                parts = line.split('->')
                src = parts[0].strip()
                dst = parts[1].strip()
                if src and dst:
                    graph.add_edge(src, dst)

parse_route_file('your_routing_file.route', net_graph)


