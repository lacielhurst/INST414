
import pandas as pd
import scipy.spatial.distance

# Load the dataset
spotify = pd.read_csv("C:\\Users\\lacie\\Lacie Files\\INST 414\\dataset.csv")

# Create a subset of the data with relevant columns
spotify_subset = spotify[["track_name", "danceability", "energy", "valence", "acousticness", "track_genre"]]

# Filter for pop genre and remove duplicates
spotify_pop = spotify_subset[spotify_subset["track_genre"] == "pop"] 
spotify_pop = spotify_pop.drop_duplicates(subset="track_name").reset_index(drop=True)
spotify_pop = spotify_pop.assign(track_name=spotify_pop["track_name"].str.lower())

# Set the track name as the index
df_tracks = spotify_pop.set_index("track_name")

target_tracks = ["cardigan", "blinding lights", "sweater weather"]

# Features for similarity
features = ["danceability", "energy", "valence"]

# Calculate similarity
for target_track in target_tracks:
    
    target_features = df_tracks.loc[target_track, features]
    
    distances = scipy.spatial.distance.cdist(df_tracks[features],[target_features], metric="euclidean")[:, 0]

    query_distances = list(zip(df_tracks.index, distances))

    top_tracks = sorted(query_distances, key=lambda x: x[1])[1:11]

    print(f"\nMost similar tracks to '{target_track}' based on danceability, energy, and valence:") 
    for track, dist in sorted(query_distances, key=lambda x: x[1])[1:11]: print(track, dist)