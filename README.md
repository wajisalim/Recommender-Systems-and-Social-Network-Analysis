# Building-a-Recommender-Systems-and-Social-Network-Analysis
This project combines two powerful analytical techniques—building recommender systems from movie rating data and conducting social network analysis using Facebook data.
# Building A Recommender 

Upload and Clean Data
```bash
# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
```
Reading data from a file into a pandas DataFrame.
```
# Read user data
u_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('/content/drive/MyDrive/IS470_data/u.user', sep='|', names=u_columns)
users
```
![image](https://github.com/wajisalim/Recommender-Systems-and-Social-Network-Analysis/assets/139387167/31fe8204-8cfc-404d-9c72-6fd8e3dd5918)

```
# Read movie data
m_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('/content/drive/MyDrive/IS470_data/u.item', sep='|', names=m_columns, usecols=range(5), encoding='latin-1')
movies
```
![image](https://github.com/wajisalim/Recommender-Systems-and-Social-Network-Analysis/assets/139387167/5ec4a93a-f092-4c9a-bec8-ef7dceaa5fe0)

```
# Read rating data
r_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('/content/drive/MyDrive/IS470_data/u.data', sep = '\t', names=r_columns)
ratings
```
![image](https://github.com/wajisalim/Recommender-Systems-and-Social-Network-Analysis/assets/139387167/98a72c86-7910-4aa8-8a63-0aacaf8f6812)

```
```
Merging the dataframe, so the dataset is ready for building the Recommender 
```
# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
MovieLense = pd.merge(movie_ratings, users)
MovieLense
```
![image](https://github.com/wajisalim/Recommender-Systems-and-Social-Network-Analysis/assets/139387167/377ade40-69cd-42e6-a07a-731e811a0cbd)
```

# Show the head of data frame
MovieLense.head()

# Rating information
MovieLense['rating'].mean()
```
Creating a bar plot to visualize the distribution of movie ratings in the MovieLense DataFrame using the Seaborn library.
```
# Rating distribution
sns.countplot(x='rating', data=MovieLense)
```
![image](https://github.com/wajisalim/Recommender-Systems-and-Social-Network-Analysis/assets/139387167/c06b0196-f393-4692-a25d-a6f7e1c2de48)
```
# Partition the data
target = MovieLense['rating']
predictors = MovieLense[['user_id', 'movie_id', 'rating']]
predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.3, random_state=0)
print(predictors_train.shape, predictors_test.shape, target_train.shape, target_test.shape)
```
Creating a matrix for user preferences on different movies, and then calculating similarities between users and movies.
```
# calculate the average rating for each user
average_user_rating = np.true_divide(train_matrix.sum(1),(train_matrix!=0).sum(1))

# create a train_matrix_sp represents users' preferences on different movies
train_matrix_sp = csr_matrix(train_matrix, dtype=np.float64)
nz = train_matrix_sp.nonzero()
train_matrix_sp[nz] -= average_user_rating[nz[0]]
train_matrix_sp = train_matrix_sp.toarray()

# calculate the user and movie similarity
user_similarity = pairwise_distances(train_matrix_sp)
movie_similarity = pairwise_distances(train_matrix_sp.T)
np.fill_diagonal(user_similarity, 0)
np.fill_diagonal(movie_similarity, 0)
print(user_similarity)
print(movie_similarity)

# Create a collaborative filtering algorithm
zero_index = np.zeros(train_matrix_sp.shape)
zero_index[nz] = 1
def collaborative_filtering (type = 'user'):
  if type == 'user':
    pre_rating = average_user_rating[:, np.newaxis] + np.dot(user_similarity, train_matrix_sp)/np.dot(user_similarity, zero_index)
  if type == 'item':
    pre_rating = (np.dot(movie_similarity, train_matrix.T)/np.dot(movie_similarity, zero_index.T)).T
  return pre_rating

# make predictions
user_prediction = collaborative_filtering(type='user')
item_prediction = collaborative_filtering(type='item')
user_prediction = np.nan_to_num(user_prediction, nan=4)
item_prediction = np.nan_to_num(item_prediction, nan=4)

# Import packages
import networkx as nx
import seaborn as sns

# Read data
G = nx.read_edgelist("/content/drive/MyDrive/IS470_data/facebook_edges.txt")
```
Social Network Analysis
```
# Visualize the network (every time you run this line of code, the network will be different)
nx.draw_networkx(G)
```
![image](https://github.com/wajisalim/Recommender-Systems-and-Social-Network-Analysis/assets/139387167/a28acd5d-8893-4281-a72a-a0f76a513a47)
```

# Number of nodes
G.number_of_nodes()

# Number of edges
G.number_of_edges()

# Degree centrality
dict(nx.degree(G)).values()
sns.histplot(dict(nx.degree(G)).values(), binwidth = 1)
```
![image](https://github.com/wajisalim/Recommender-Systems-and-Social-Network-Analysis/assets/139387167/c7fa1603-dc0b-4131-9355-306055a0a7b0)

```
# Betweenness centrality
nx.betweenness_centrality(G, normalized=False)

# Closeness centrality
nx.closeness_centrality

# Transitivity
nx.transitivity(G)

# plot the largest cliques (from https://orbifold.net/default/community-detection-using-networkx/)
cliques = list(nx.find_cliques(G))
max_clique = max(cliques, key=len)
node_color = [(0.5, 0.5, 0.5) for v in G.nodes()]
for i, v in enumerate(G.nodes()):
  if v in max_clique:
    print(v)
    node_color[i] = (0.9, 0.5, 0.5)
nx.draw_networkx(G, node_color=node_color)
```
![image](https://github.com/wajisalim/Recommender-Systems-and-Social-Network-Analysis/assets/139387167/0735330f-2804-4d4e-98a1-90462978effc)

```

# Community detection (from https://orbifold.net/default/community-detection-using-networkx/)
def get_color(i, r_off=1, g_off=1, b_off=1):
  r0, g0, b0 = 0, 0, 0
  n = 16
  low, high = 0.1, 0.9
  span = high - low
  r = low + span * (((i + r_off) * 3) % n) / (n - 1)
  g = low + span * (((i + g_off) * 5) % n) / (n - 1)
  b = low + span * (((i + b_off) * 7) % n) / (n - 1)
  return (r, g, b)  
def set_node_community(G, communities):
  for c, v_c in enumerate(communities):
    for v in v_c:
      G.nodes[v]['community'] = c + 1

result = nx.community.girvan_newman(G)
communities = next(result)
set_node_community(G, communities)
node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]

nx.draw_networkx(G, node_color=node_color)
```
![image](https://github.com/wajisalim/Recommender-Systems-and-Social-Network-Analysis/assets/139387167/196d1496-f51a-4d55-933a-d1a00448c5ae)

```




