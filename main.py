from fastapi import FastAPI, HTTPException
import pandas as pd 
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




# Cargo el archivo Parquet en un DataFrame
df = pd.read_parquet('dataset_transformado.parquet')
# Limpio algunos datos
df = df.dropna(subset=['title', 'vote_average'])
df['vote_average'] = df['vote_average'].fillna(0)

app = FastAPI()

class DescriptionInput(BaseModel):
    description: str

# Endpoint de cantidad de filmaciones en un mes
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    meses = {'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12}
    mes_num = meses.get(mes.lower())
    if not mes_num:
        raise HTTPException(status_code=400, detail="Mes no válido")
    
    cantidad = df[df['release_date'].dt.month == mes_num].shape[0]
    return {"mes": mes, "cantidad_filmaciones": cantidad}


# Endpoint para buscar por titulo, tambien devuelve el score de esa pelicula
@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    pelicula = df[df['title'].str.contains(titulo, case=False, na=False)]
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    # Codigo que ya no uso
    #if len(pelicula) > 1:
        raise HTTPException(status_code=400, detail="Existen múltiples películas con ese título, especifica mejor el título.")
    
    score = pelicula['vote_average'].iloc[0]
    return {"titulo": titulo, "score": score}


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['title'].fillna(''))
# Funcion de recomendacion
def get_similar_movies(titulo: str) -> List[str]:
    # Me aseguro de que el titulo este en minusculas para compararlo
    titulo = titulo.lower()
    
    # Verifico si la película existe
    idx = df[df['title'].str.contains(titulo, case=False, na=False)].index
    if not idx.size:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    
    idx = idx[0]
    # Obtengo el indice de la pelicula en el df
    pelicula_tfidf = tfidf_matrix[idx]
    
    # Calculo similitudes
    cosine_sim = cosine_similarity(pelicula_tfidf, tfidf_matrix)
    
    # Obtengo índices de las películas similares
    similar_indices = cosine_sim.argsort().flatten()[::-1]
    
    # Selecciono las top 5 películas similares
    similar_indices = similar_indices[similar_indices != idx][:5]
    similar_movies = df.iloc[similar_indices]
    
    # Extraigo títulos de las películas recomendadas
    return similar_movies['title'].tolist()

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    similar_movies = get_similar_movies(titulo)
    return {"titulo": titulo, "recomendaciones": similar_movies}



# Ejecuto la App

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

