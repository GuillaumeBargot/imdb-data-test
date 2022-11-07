#!/usr/bin/env python
# coding: utf-8

# # Importation des librairies et création des dataframes

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
import time
import pandas as pd
import sqlite3

from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime, timezone


# In[3]:


# Create your connection.
cnx = sqlite3.connect('data/movies.sqlite')

movies_df = pd.read_sql_query("SELECT * FROM movies", cnx)
ratings_df = pd.read_sql_query("SELECT * FROM ratings", cnx)

cnx.commit()
cnx.close()


# In[4]:


movies_df.head()


# In[5]:


ratings_df.head()


# J'ai choisi d'utiliser pandas pour répondre aux questions du test car c'est l'outil que j'ai le plus utilisé lors de ma formation.

# # 1. Dénombrements

# ## 1.1 Combien de films figurent dans la base de données ?

# In[6]:


len(movies_df.movie_id.unique())


# In[7]:


len(movies_df)


# On peut voir qu'il y a 5 films de plus qu'il y a de films unique. Regardons quel est le problème avec ces films.

# In[8]:


movies_df.loc[movies_df.movie_id.duplicated(keep=False)]


# On supprime une entrée au hasard pour les deux films complétement identiques à leur doublon (Tom et Jerry et Toy Story 4)

# In[9]:


movies_df = movies_df.drop_duplicates()


# Pour les 3 autres films avec des doublons, nous allons supprimer manuellement la moins bonne ligne:
#     
#     - pour Carlito's Way, il vaut garder la ligne comprenant le genre
#     
#     - pour Don't Breathe, il vaut mieux garder la ligne qui ne comporte pas de faute
#     
#     - pour Promising Young Woman il vaut mieux garder la ligne comprenant le genre également.
#     
# S'il y avait eu plus de 3 films, il aurait fallu trouver une solution systèmatique pour chacun de ces problème, en faire une fonction qui permetterai de choisir la bonne ligne à garder.

# In[10]:


movies_df = movies_df.drop([8456, 29746, 36642])


# On recompte le nombre de films:

# In[11]:


print(len(movies_df.movie_id.unique()))
print(len(movies_df))
nb_movies = len(movies_df)


# Nous allons créer une fonction pour entrer les résultats dans le fichier texte.

# In[12]:


def writeTitle(string):
    with open("results.txt", "a", encoding="utf8") as f:
        f.write('\n--------------------------------\n')
        f.write(string + '\n')
        f.write('--------------------------------\n')
        
def writeLine(string):
    with open("results.txt", "a", encoding="utf8") as f:
        f.write(string + '\n')


# Premièrement, j'éfface ce qu'il y a dans le fichier texte pour pouvoir le réécrire à chaque exécution de ce notebook.

# In[13]:


with open("results.txt", "w", encoding="utf8") as f:
        f.write('')


# In[14]:


writeTitle("1.1 Combien de films figurent dans la base de données ?")
writeLine(f"{nb_movies} films figurent dans la base de données.")


# ## 1.2. Combien d'utilisateurs différents figurent dans la base de données ?

# Ici, on compte directement le nombre d'id d'utilisateurs uniques.

# In[15]:


different_users = len(ratings_df.user_id.unique())
print(different_users)


# In[16]:


writeTitle("1.2. Combien d'utilisateurs différents figurent dans la base de données ?")
writeLine(f"{different_users} utilisateurs différents figurent dans la base de données.")


# ## 1.3. Quelle est la distribution des notes renseignées ?

# On compte le nombre de rating avec une note donnée, on en fait un tableau, puis un graph.

# In[17]:


ratings_table = ratings_df.value_counts('rating').sort_index()


# In[18]:


ratings_table


# In[19]:


plt.title('Histogramme de la distribution des notes IMDB')
plt.xlabel("Note")
plt.ylabel("Quantité")
ratings_df.rating.value_counts().sort_index().plot(kind = 'bar')


# In[20]:


writeTitle("1.3. Quelle est la distribution des notes renseignées ?")
writeLine("Distribution des notes IMDB:\n" + str(ratings_table))


# ## 1.4. Table des fréquences pour exprimer en pourcentage la répartition des notes

# Ici, on créé une nouvelle colonne ou l'on divise le nombre de rating de telle note par le nombre de rating total.

# In[21]:


ratings_frequency_table = ratings_df[['rating', 'user_id']].groupby(by="rating").count()


# In[22]:


ratings_frequency_table = ratings_frequency_table.rename(columns={'user_id':'frequency'})


# In[23]:


ratings_frequency_table['frequency'] = ratings_frequency_table['frequency']/len(ratings_df)


# In[24]:


ratings_frequency_table


# On vérifie que la somme des fréquences est égale à 1:

# In[25]:


ratings_frequency_table.frequency.sum()


# In[26]:


writeTitle("1.4. Table des fréquences pour exprimer en pourcentage la répartition des notes")
writeLine("Tables des fréquences des notes:\n" + str(ratings_frequency_table))


# # 2. Sélection et enrichissement des données

# ## 2.1.

# On créé une fonction qui décide si l'utilisateur a aimé le film ou non via le rating.

# In[27]:


def isLikedRating(rating):
    if(rating > 6):
        return 1
    else:
        return 0


# In[28]:


ratings_df['liked'] = ratings_df.apply(lambda x: isLikedRating(x.rating), axis=1)


# In[29]:


ratings_df


# In[30]:


writeTitle("2.1. Transformer la note rating en deux modalités : l'utilisateur a-t-il aimé ou pas le film ?")
writeLine("10 premières lignes de la table ratings avec les \"like\"\n" + str(ratings_df.head(10)))


# ## 2.2

# Il y a potentiellement deux questions dans ce 2.2 : "Quels sont les genres les mieux notés par les utilisateurs ?" qui demanderait le rating moyen de chaque genre de film et "Nous souhaitons obtenir le top 10 des genres de films aimés par les utilisateurs" qui nous demanderait de comptabiliser le nombre de "liked" qu'on a créé dans la question précédente. Par continuité de l'exercice, je pense que c'est la deuxième interprétation qui est la bonne mais je souhaite quand même répondre aux deux.

# Nous allons tout d'abord créer differentes colonnes dans la table movies pour séparer les genres entre eux.

# In[31]:


movies_df.genre.unique()


# In[32]:


pd.DataFrame(movies_df['genre'].str.split('|', expand=True).values)


# Bien que cette façon de faire marche, dans le sens ou cela nous donne differents genres de films pour chaque film, cela nous demande plus d'étapes pour ensuite savoir les notes des films d'un genre en particulier.

# Au lieu de ça, je vais créer une fonction qui me créera autant de colonnes qu'il y a de genre. Cela abouti à un tableau avec beaucoup plus de colonnes mais qui sera plus facile à utiliser par la suite.

# Cela équivaut à faire du one-hot encoding mais avec plusieurs labels. Avec Sklearn, on peut utiliser la fonction MultiLabelBinarizer

# In[34]:


list_of_genres = []
def getListOfGenre(genreString):
    if pd.notnull(genreString):
        return genreString.split('|')
    return ['None']

def cleanMovieGenre(df):
    df['genre'] = df.apply(lambda x: getListOfGenre(x.genre), axis=1)
    return df

def multiLabelBinarizeGenres(df):
    global list_of_genres
    mlb = MultiLabelBinarizer()
    mlb.fit(df['genre'])
    list_of_genres = mlb.classes_

    new_col_names = ["%s" % genre for genre in mlb.classes_]

    # Create new DataFrame with transformed/one-hot encoded IDs
    ids = pd.DataFrame(mlb.fit_transform(df['genre']), columns=new_col_names)
    print("Len ID: " + str(len(ids)))
 
    # Concat with original `Label` column
    return pd.concat( [df[['movie_id', 'title']].reset_index(drop=True), ids], axis=1)

def createGenreColumns():
    global movies_df
    movies_df = cleanMovieGenre(movies_df)
    movies_df = multiLabelBinarizeGenres(movies_df)


# In[35]:


createGenreColumns()


# In[36]:


movies_df


# Maintenant, on peut créer un dataframe avec les moyenne des notes de chaque genre

# In[37]:


def getMoviesOfGenre(genre):
    return movies_df.loc[movies_df[genre] == 1]

def joinMoviesWithRating(genre):
    movies_of_genre = getMoviesOfGenre(genre)
    return ratings_df.merge(movies_of_genre,on='movie_id')

def getMeanRatingOfGenre(df):
    return np.mean(df.rating)

def getNumberOfLikesOfGenre(df):
    return np.sum(df.liked)

def createAverageRatingTableForGenre():
    list_of_averages = []
    list_of_likes = []
    for genre in list_of_genres:
        joined_table = joinMoviesWithRating(genre)
        list_of_averages.append(getMeanRatingOfGenre(joined_table))
        list_of_likes.append(getNumberOfLikesOfGenre(joined_table))
    return pd.DataFrame(zip(list_of_genres,list_of_averages, list_of_likes), columns=['genre', 'average_rating', 'number_of_likes'])


# In[38]:


rating_table = createAverageRatingTableForGenre()
rating_table


# In[39]:


top10_genre_by_ratings = rating_table.sort_values(by='average_rating', ascending=False).head(10)
top10_genre_by_ratings


# Maintenant, on pourrait effectuer plusieurs opérations sur ce classement, tel qu'enlever la catégorie "None" qui représente tous les films ou le genre n'était pas indiqué dans notre base de données. On pourrait aussi mettre en place une sorte de threshold de nombre de ratings qui empêcherai des genres très peu représentés mais avec quelques bonnes notes de figurer sur notre top10.

# In[40]:


top10_genre_by_likes = rating_table.sort_values(by='number_of_likes', ascending=False).head(10)
top10_genre_by_likes


# In[41]:


writeTitle("2.2 Nous souhaitons obtenir le top 10 des genres de films aimés par les utilisateurs")
writeLine("Top 10 des genres par nombre de \"like\"\n" + str(top10_genre_by_likes))


# # Sélections avancées

# ## 3.1 Quels sont les titres les plus aimés des internautes ?

# On groupe les ratings par film, en transformant les colonnes de rating en moyenne et on somme les nombre de like. On utilise aussi la colonne de rating_timestamp pour compter le nombre de rating qu'un film a reçu.

# In[42]:


def createRatingsGroupedByMovie(ratings_df):
    ratings_by_movie = ratings_df.groupby('movie_id').agg({'rating':'mean', 'liked':'sum','rating_timestamp':'count'}).rename(columns={'rating_timestamp':'count'})
    return movies_df[['movie_id', 'title']].merge(ratings_by_movie, on='movie_id')


# In[43]:


ratings_by_movie = createRatingsGroupedByMovie(ratings_df)
ratings_by_movie


# In[44]:


ratings_by_movie = ratings_by_movie.loc[ratings_by_movie['count'] >= 5]
ratings_by_movie


# Ici, je choisi de faire un classement en utilisant en priorité la note moyenne. Mais si il y a égalité, j'utilise ensuite le nombre de rating considéré comme "liked" et enfin le nombre de rating lui même.

# In[45]:


top10_movies_by_ratings = ratings_by_movie.sort_values(by=['rating', 'liked', 'count'], ascending=False).head(10)
top10_movies_by_ratings


# In[46]:


writeTitle('3.1 Quels sont les titres les plus aimés des internautes ?')
writeLine('Tableau des 10 films les plus aimés des internautes:\n' + str(top10_movies_by_ratings[['title', 'rating', 'liked', 'count']]))


# ## 3.2 Quel est le film le plus noté durant l'année 2000 ?

# Créons une fonction qui retourne une fourchette de temps Unix qui correspond à une année:

# In[47]:


def getUnixYearTimeSpan(year):
    begin = datetime(year, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(year+1, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    return (begin.timestamp(), end.timestamp())


# In[48]:


getUnixYearTimeSpan(2020)


# In[49]:


def getMovieRatingsForYear(year):
    yearSpan = getUnixYearTimeSpan(year)
    return ratings_df.loc[(ratings_df.rating_timestamp > yearSpan[0]) & (ratings_df.rating_timestamp < yearSpan[1])]


# In[50]:


ratings_2020 = getMovieRatingsForYear(2020)
ratings_2020


# Maintenant, on peut refaire notre table de la question précedante, qui contenait le nombre de ratings.

# In[51]:


movie_rated_2020 = createRatingsGroupedByMovie(ratings_2020).sort_values('count', ascending=False)
movie_rated_2020


# Le film le plus noté de 2020 est donc:

# In[52]:


movie_rated_2020.head(1)


# In[53]:


writeTitle("3.2 Quel est le film le plus noté durant l'année 2020 ?")
writeTitle("Le film le plus noté durant l'année 2020 est : \n" + str(movie_rated_2020.head(1)))


# 1917 !

# # Gestion des données

# On créé une fonction qui execute la requête voulue et qui chronomètre le temps que l'on prends pour la faire 100 fois. (nous permet d'avoir des temps en secondes et des différences plus marquées)

# In[55]:


start = time.time()
print("hello")
end = time.time()
print(end - start)


# In[56]:


def testRequest():
    # Create your connection.
    cnx = sqlite3.connect('data/movies.sqlite')
    start = time.time()
    for i in range(100):
        request = pd.read_sql_query('SELECT * FROM ratings WHERE user_id == 255', cnx)
    cnx.commit()
    cnx.close()
    end = time.time()
    return str(end-start) + 's'


# In[57]:


request_time_without_index = testRequest()
request_time_without_index


# In[58]:


def indexUserId():
    cnx = sqlite3.connect('data/movies.sqlite')
    cur = cnx.cursor()
    cur.execute('CREATE INDEX user_id_asc ON ratings(user_id ASC)')
    cnx.commit()
    cnx.close()


# In[59]:


indexUserId()


# In[60]:


request_time_with_index = testRequest()
request_time_with_index


# In[61]:


def dropIndex():
    cnx = sqlite3.connect('data/movies.sqlite')
    cur = cnx.cursor()
    cur.execute('DROP INDEX user_id_asc')
    cnx.commit()
    cnx.close()


# In[62]:


dropIndex()


# In[64]:


writeTitle('Gestion des données')
writeLine('Temps de 100 requêtes sans indexer user_id: \n' + request_time_without_index)
writeLine('Temps de 100 requêtes en indexant user_id: \n' + request_time_with_index)


# On voit une différente assez importante (8 secondes contre 5 centièmes de secondes) quand on éxecute 100 fois la requête. C'est donc 160 fois plus rapide en utilisant ces fonctions. En effet, il est possible que ma façon de multiplier les requêtes entraîne des temps supplémentaires à des endroits par rapport à une utilisation sans boucle dans une situation réelle.

# Le fait d'indexer une colonne la rends plus rapide à parcourir, typiquement pour des requête WHERE column = value car on créé des sortes de pointeurs et en les utilisant au lieu de parcourir chaque ligne de la base de données.
