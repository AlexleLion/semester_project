import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_author_title(file_path):
    # Extraire tous les noms du fichier
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    title = file_path.split('/')[4]
    
    author_pattern = re.compile(r'\b(Mme|Madame|M|Mr|Dr|Docteur|docteur|Monsieur|monsieur|madame|professeur|Professeur|Comtesse)\b\s?[A-Z][a-zA-Z]*\s?[A-Z]?[a-zA-Z]*')
    author_match = author_pattern.search(content[:500])
    author = author_match.group(0) if author_match else 'auteur inconnu'
    
    return title, author

def load_authors(directory_path, output_csv):
    # Créer une liste de listes pour stocker les données des personnes citées
    data_mixtes = []

    for subdirectory in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory)
        if os.path.isdir(subdirectory_path):

            num_pages = len([filename for filename in os.listdir(subdirectory_path)])

            for filename in os.listdir(subdirectory_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(subdirectory_path, filename)
                    title, author = extract_author_title(file_path)
                    if author != 'auteur inconnu':
                        data_mixtes.append([title, author, num_pages])

    df = pd.DataFrame(data_mixtes, columns=['Titre', 'Auteur', 'Nbr_pages'])
    df['Auteur'] = df['Auteur'].str.replace('\n', ' ')  
    df.to_csv(output_csv, index=False)


def load_year(directory, output_csv):
    # Récupère les années de publication de chaque ouvrage
    data = []

    for subdirectory in os.listdir(directory):
        subdirectory_path = os.path.join(directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            for filename in os.listdir(subdirectory_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(subdirectory_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    title = file_path.split('/')[4]
                    years = re.findall(r'\b(18[0-9]{2}|19[0-9]{2})', content)
                    if years:
                        data.append([title, ', '.join(years)])

    df_years = pd.DataFrame(data, columns=['Titre', 'Années'])
    df_years = df_years.groupby("Titre")["Années"].apply(lambda x: ', '.join(sorted(set(', '.join(x).split(', '))))).reset_index()
    df_years.to_csv(output_csv, index=False)

load_authors("../data/txt/ouvrages_mixtes", "../data/contexte/auteurs_mixtes.csv")
load_authors("../data/txt/ouvrages_feminins", "../data/contexte/auteurs_feminins.csv")
load_authors("../data/txt/ouvrages_masculins", "../data/contexte/auteurs_masculins.csv")

load_year("../data/txt/ouvrages_mixtes", "../data/contexte/years_pub_mixtes.csv")
load_year("../data/txt/ouvrages_feminins", "../data/contexte/years_pub_feminins.csv")
load_year("../data/txt/ouvrages_masculins", "../data/contexte/years_pub_masculins.csv")

df_fem = pd.read_csv("../data/contexte/years_pub_feminins.csv")
df_mas = pd.read_csv("../data/contexte/years_pub_masculins.csv")
df_mix = pd.read_csv("../data/contexte/years_pub_mixtes.csv")

