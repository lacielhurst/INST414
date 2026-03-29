
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset in
dogs = pd.read_csv("c:\\Users\\lacie\\Lacie Files\\INST 414\\dogs_ranking.csv")

# Select relevant columns and rename them
dogs = dogs[["Breed","$LIFETIME COST", "INTELLIGENCE %", "NUMBER OF GENETIC AILMENTS", "LONGEVITY(YEARS)", "size", "score for kids"]]
dogs = dogs.rename(columns={"$LIFETIME COST": "lifetime cost", 
                            "INTELLIGENCE %": "intelligence percent", 
                            "NUMBER OF GENETIC AILMENTS": "number of genetic ailments",
                            "LONGEVITY(YEARS)": "longevity",
                            "Breed": "breed"})

# Clean the data by removing symbols and converting to appropriate data types
dogs["lifetime cost"] = dogs["lifetime cost"].str.replace("$", "").str.replace(",", "").astype(float)
dogs["intelligence percent"] = dogs["intelligence percent"].str.replace("%", "").astype(float)

# Create a new column for size categories
size_map = {1: "Small", 2: "Medium", 3: "Large"}
dogs["size_char"] = dogs["size"].map(size_map)

# Visualizations
custom_palette = {
    "Small": "indianred",
    "Medium": "cadetblue",
    "Large": "darkslateblue"
}
plt.figure(figsize=(12, 6))
sns.set_style("darkgrid")
sns.scatterplot(data=dogs, x='lifetime cost', y='longevity', hue = "size_char", alpha=0.9, palette=custom_palette)
plt.title("Lifetime Cost vs Longevity for Dogs by Size")
plt.xlabel("Lifetime Cost ($)")
plt.ylabel("Longevity (Years)")
plt.legend(title="Dog Size")

cheap_dogs = dogs.sort_values("lifetime cost").head(20)
plt.figure(figsize=(12,6))
sns.set_style("darkgrid")
sns.barplot(data=cheap_dogs.sort_values("lifetime cost", ascending=False), x="breed", y="lifetime cost", color = "goldenrod")
plt.xticks(rotation=45, fontsize=8, ha = 'right')
plt.xlabel("Breed")
plt.ylabel("Lifetime Cost ($)")
plt.title("Lifetime Cost for the 20 Cheapest Dog Breeds")
plt.tight_layout()

plt.figure(figsize=(12, 6))
sns.set_style("darkgrid")
sns.scatterplot(data=dogs, x='intelligence percent', y='score for kids', alpha=0.9, color = "steelblue")
plt.title("Dog Intelligence vs Suitability for Families with Children")
plt.xlabel("Intelligence (%)")
plt.ylabel("Suitability for Families with Children")

risk_dogs = dogs.sort_values("number of genetic ailments", ascending=False).head(20)
plt.figure(figsize=(12,6))
sns.set_style("darkgrid")
sns.barplot(data=risk_dogs.sort_values("number of genetic ailments", ascending=False), x="breed", y="number of genetic ailments", color = "darkolivegreen")
plt.xticks(rotation=45, fontsize=8, ha = 'right')
plt.ylabel("Number of Genetic Ailments")
plt.xlabel("Breed")
plt.title("Number of Genetic Ailments for the 20 Riskiest Dog Breeds")
plt.tight_layout()

plt.show()







