
import pandas as pd
import re
import networkx as nx

# Load the dataset
huffpost_data = pd.read_json("C:/Users/lacie/Downloads/News_Category_Dataset_v3.json", lines=True)

# Function to clean the authors string and return a list of authors
def clean_authors(author_string):

    if pd.isna(author_string) or author_string.strip() == "":
        return []
    
    # Remove ', AP'
    author_string = re.sub(r",\s*AP$", "", author_string, flags=re.IGNORECASE)
    
    # Split on ' and ' or commas
    authors = re.split(r"\s+and\s+|,", author_string)
    
    # Strip whitespace and remove empty entries 
    authors = [a.strip() for a in authors if a.strip() != ""]
    
    return authors

# Function to filter out generic roles and keep likely real names
def filter_real_authors(authors):
   
    generic_roles = {"ContributorExecutive Director", "ContributorSenior Fellow"}
    
    filtered = []
    for a in authors:
        a_clean = a.strip()
        # keep only if not a generic role AND has at least one space (likely a real name)
        if a_clean not in generic_roles and " " in a:
            filtered.append(a.strip())
    return filtered

# Clean the raw strings into lists
huffpost_data["author_list"] = huffpost_data["authors"].apply(clean_authors)

# Filter out generic roles
huffpost_data["author_list"] = huffpost_data["author_list"].apply(filter_real_authors)

# Create a subset of the data for the "POLITICS" category
df_subset = huffpost_data[huffpost_data["category"] == "POLITICS"]

# Create a graph where nodes are authors and edges represent co-authorship
g = nx.Graph()

# Iterate through the author lists and add edges for co-authorship
for author_list in df_subset["author_list"]:
    if len(author_list) == 0:
        continue
    for author_name in author_list:
        g.add_node(author_name)
    i = 0
    for left_author in author_list:
        for right_author in author_list[i+1:]:
            current_weight = g[left_author][right_author]['weight'] if g.has_edge(left_author, right_author) else 0
            g.add_edge(left_author, right_author, weight=current_weight + 1)
        i += 1

# Print top 20 authors by degree
deg_list = sorted(g.degree, key=lambda x: x[1], reverse=True)[:20]
print("\nTop 20 authors by Number of Co-Authors:")
for author_name, deg in deg_list:
    print(f"{author_name} - {deg}")

# Remove edges with weight less than 2 and any resulting isolated nodes
edges_to_remove = [(u, v) for u, v, d in g.edges(data=True) if d['weight'] < 2]
g.remove_edges_from(edges_to_remove)
g.remove_nodes_from(list(nx.isolates(g))) 

# Export the graph to GraphML format
nx.write_graphml(g, "authors.graphml")


