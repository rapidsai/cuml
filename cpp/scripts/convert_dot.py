import pydot
import os
import typing
import shutil
import glob
from stat import S_IWUSR, S_IREAD, S_IRGRP, S_IROTH
import uuid
import io

root_dir = "/home/mdemoret/Repos/rapids/cuml-dev"
cpp_dir = os.path.join(root_dir, "cpp")


def process_dot_file(src_dot):

    dot_abs_path = os.path.join(root_dir, src_dot)
    dot_obj: pydot.Dot = None

    # First determine the backup path
    dot_abs_path_split = os.path.splitext(dot_abs_path)
    dot_abs_backup_path = dot_abs_path_split[0] + "_OLD" + dot_abs_path_split[1]

    # If the backup path exists, load that
    if (os.path.exists(dot_abs_backup_path)):
        # We have processed this before
        dot_obj = pydot.graph_from_dot_file(dot_abs_backup_path)[0]
    else:
        # Have not processed this before. Load and backup
        dot_obj = pydot.graph_from_dot_file(dot_abs_path)[0]

        shutil.move(dot_abs_path, dot_abs_backup_path)

    # Get the included header name
    name = dot_obj.get_name().strip('"').replace(cpp_dir + "/", "")

    nodes: typing.List[pydot.Node] = dot_obj.get_nodes()

    nodes_dirs: typing.Dict[str, typing.List[pydot.Node]] = {}

    compilation_count = 0

    for n in nodes:
        node_name: str = n.get("label")

        # Set the default node shape to be box instead of record
        if (n.get_name() == "node"):
            n.set("shape", "box")

        if (node_name is None):
            continue

        node_name = node_name.strip('"')

        node_name = node_name.replace("\\l", "")

        # Remove the root path
        node_name = node_name.replace(cpp_dir + "/", "")

        n.set("label", '"' + node_name + '"')

        node_dir = node_name.split("/")[0]

        if (node_dir not in nodes_dirs):
            nodes_dirs[node_dir] = []

        nodes_dirs[node_dir].append(n)

        # Now determine if the node is a compilation unit
        node_ext = os.path.splitext(node_name)[1]

        if (node_ext == ".cpp" or node_ext == ".cu"):
            n.set("fillcolor", "#ffeeee")
            n.set("color", "red")

            compilation_count += 1

    for dir, dir_nodes in nodes_dirs.items():

        dir_sub_graph = pydot.Subgraph(graph_name="cluster" + dir)

        # dir_sub_graph.add_node(pydot.Node(name="graph", bgcolor="#eeeeff", pencolor="black"))
        dir_sub_graph.set_graph_defaults(bgcolor="#eeeeff", pencolor="black")

        dir_sub_graph.add_node(pydot.Node(name=dir, shape="plaintext"))

        dot_obj.add_subgraph(dir_sub_graph)

        for n in dir_nodes:

            dot_obj.del_node(n.get_name())

            dir_sub_graph.add_node(n)

    svg_abs_path = dot_abs_path_split[0] + ".svg"
    md5_abs_path = dot_abs_path_split[0] + ".md5"
    map_abs_path = dot_abs_path_split[0] + ".map"

    # Ensure we can write the file
    if (os.path.exists(dot_abs_path)):
        os.chmod(dot_abs_path, S_IWUSR | S_IREAD)

    # Write the new dot file
    dot_obj.write(dot_abs_path)

    # Make this write only (Preventing doxygen from reverting this)
    os.chmod(dot_abs_path, S_IREAD | S_IRGRP | S_IROTH)

    # Now write a random hash to the .md5 file to force doxygen to re-patch the
    # output files
    with io.open(md5_abs_path, mode='w') as f:
        f.write(uuid.uuid4().hex)

    # Output the SVG
    dot_obj.write(svg_abs_path, prog="dot", format="svg")

    # Output the map file
    dot_obj.write(map_abs_path, prog="dot", format="cmapx")

    return name, compilation_count


def main():

    glob_pattern = "cpp/build/html/*__dep__incl.dot"

    # glob_pattern = os.path.join(glob_dir, glob_file_pattern)

    dot_files = glob.glob(glob_pattern)

    comp_counts = {}

    for f in dot_files:

        name, comp_count = process_dot_file(f)

        comp_counts[name] = comp_count

        # if (len(comp_counts) >= 50):
        #     break

    for k, v in sorted(comp_counts.items(), key=lambda item: -item[1]):
        print("File: {}, Comp Count: {}".format(k, v))

    print("Done. Now run `./build.sh cppdocs` to update the documentation")


if __name__ == "__main__":
    main()
