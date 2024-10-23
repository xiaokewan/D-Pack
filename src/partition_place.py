def parse_place_file(file_path):
    with open(file_path, 'r') as file:
        next(file) 
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.split()
            block_name = parts[0]
            x = int(parts[1])
            y = int(parts[2])

            if block_name in net_graph:
                net_graph.nodes[block_name]['pos'] = (x, y)

parse_place_file('your_placement_file.place')
