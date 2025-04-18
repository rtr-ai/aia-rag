# Interactive Network Graph Visualization of AiAct

![Screenshot 2024-12-03 at 16-49-44 (PNG Image 2524 × 866 pixels)](https://github.com/user-attachments/assets/3047de59-2d65-4944-86ee-32ddf06a8f59)


### Introduction

Each node represents a single chunk of textual data that has been fed into the vector store. The process of chunking the data was largely manual to achieve an optimal structure for the embeddings. Each chunk corresponds to a specific unit, such as a chapter (Kapitel), section (Abschnitt), article (Artikel), or similar. Additionally, each chunk contains information about its relationship to other chunks (Querverweise).

In the visualization, each node represents an individual embedding (e.g., chapter, section), while the connecting lines between nodes (attributes) depict the relationships between two or more nodes.

### How to open interactive network visualization

Download the .html file from this folder and save it locally on your computer. Open the file in a web browser application, and the visualization will be displayed.

### How to use the Interactive Graph

You can zoom in and out to get a clearer view of the node names. Hover your mouse over the nodes or attributes to see the text appear. By holding the left mouse button on a node, you can drag and move it across the visualization.

The size of the node represents the number of edges, or connections, it has with its neighboring nodes.

#### Filter

There are several ways to filter the results of the graph. You can filter by nodes, edges, and their properties (such as color, size, title, ID, etc.), as well as by property values.

#### Physics

Here, you can adjust various settings related to the visualization itself, allowing you to configure the level of interactivity for the network graph.
