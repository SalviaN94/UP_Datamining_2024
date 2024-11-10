import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend.frequent_patterns
import mlxtend.preprocessing
import numpy
import warnings

from textblob import TextBlob
from wordcloud import WordCloud

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

#Segmentación de clientes
data = pd.read_csv("./dataset/coffee-shop-sales-revenue.csv", delimiter='|')
data['datetime'] = pd.to_datetime(data['transaction_date'] + ' ' + data['transaction_time'])
data.head()

# Obtengo las ubicaciones de los locales
locales = data.drop_duplicates(subset=['store_location'])['store_location']
print(locales)

# Extraigo las transacciones por locales
datos_por_local = []
for local in locales:
    ventas = data[data['store_location'] == local][['datetime', 'product_type']]
    datos_por_local.append(ventas)

# Extraigo los tipos de productos que se vendieron en cada local
tmpVentas = []
productos = []
for x in range(len(datos_por_local)):
    tmpVentas.append(datos_por_local[x])
    productos.append(pd.DataFrame(datos_por_local[0]).drop_duplicates('product_type')['product_type'].tolist())    
    productos[x].insert(0, 'datetime')

# Convierto los productos en columnas
MBA_data = []
for x in range(len(productos)):
    MBA_data.append(pd.DataFrame([productos[x]]))
    MBA_data[x].columns = productos[x]

# NO EJECUTAR A MENOS QUE SEA NECESARIO
# Construyo la matriz en base a productos comprados por fecha

# Crear una matriz de "one-hot encoding" usando pivot y fillna para evitar el bucle
for x in range(len(tmpVentas)):
    # Crear una columna 'comprado' con valor 1 para indicar que se compró el producto en esa transacción
    tmpVentas[x]['comprado'] = 1

    # Convertir los datos al formato de "one-hot encoding" usando pivot
    MBA_data[x] = tmpVentas[x].pivot_table(
        index='datetime',
        columns='product_type',
        values='comprado',
        fill_value=0
    ).reset_index()

# Ahora MBA_data[x] debería tener un formato de "one-hot encoding" con menos costo computacional

# Agrupamos las columnas que tienen la misma fecha
MBA_data_grp = []
for x in range(len(MBA_data)):
    MBA_data_grp.append(MBA_data[x].groupby('datetime').sum().reset_index())
    MBA_data_grp[x] = MBA_data_grp[x].drop(columns=['datetime'])
    MBA_data_grp[x] = MBA_data_grp[x].map(lambda x: True if x >= 1 else False)

# Realizamos el análisis
frequent_itemsets = []
rules = []
for x in range(len(MBA_data_grp)):
    frequent_itemsets.append(apriori(MBA_data_grp[x], min_support=0.001, use_colnames=True))
    rules.append(association_rules(frequent_itemsets[x], metric="lift"))

# Mostramos los valores obtenidos
for x in range(len(rules)):
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"Localidad: {locales.iloc[x]}")
    rules[x].sort_values(['support', 'confidence', 'lift'], axis = 0, ascending = False).head(10)


# Arbol de decisión

# Cargar el dataset con el delimitador correcto
ruta_dataset = "./dataset/coffee-shop-sales-revenue.csv"
ventas_producto = pd.read_csv(ruta_dataset, delimiter='|')

# Ver las primeras filas del dataset para confirmar que cargó correctamente
display(ventas_producto.head())

# Extraer el mes y el año de la columna 'transaction_date'
ventas_producto['transaction_date'] = pd.to_datetime(ventas_producto['transaction_date'])
ventas_producto['Mes'] = ventas_producto['transaction_date'].dt.month
ventas_producto['Año'] = ventas_producto['transaction_date'].dt.year

# Verificar que se añadieron las nuevas columnas
display(ventas_producto[['transaction_date', 'Mes', 'Año']].head())

# Especificar el product_type y product_category de interés
producto_interes = 'Scone'  # Cambiar al producto que te interese
categoria_interes = 'Bakery'  # Cambiar a la categoría de interés

# Filtrar las ventas de este producto y categoría
ventas_filtradas = ventas_producto[(ventas_producto['product_type'] == producto_interes) &
                                   (ventas_producto['product_category'] == categoria_interes)]

# Resumir las ventas por mes para los primeros 4 meses
ventas_resumidas = ventas_filtradas.groupby('Mes')['transaction_qty'].sum().loc[1:4]
print(f"Ventas de {producto_interes} en los primeros meses:\n{ventas_resumidas}")

# Mostrar las ventas en los primeros 4 meses
for mes, cantidad in ventas_resumidas.items():
    print(f"En el mes {mes} se vendieron {cantidad} unidades de {producto_interes} ({categoria_interes}).")

# Ignorar advertencias temporales de FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Gráfico de barras para ventas por mes
plt.figure(figsize=(10, 6))
sns.barplot(x=ventas_resumidas.index, y=ventas_resumidas.values, palette="Blues", hue=None)
plt.title(f"Ventas de {producto_interes} en los primeros 4 meses")
plt.xlabel("Mes")
plt.ylabel("Cantidad de ventas")
plt.show();

# Gráfico de línea para ventas por mes y año
plt.figure(figsize=(12, 6))
sns.lineplot(data=ventas_filtradas, x="Mes", y="transaction_qty", hue="Año", marker="o", palette="tab10")
plt.title(f"Distribución de ventas de {producto_interes} ({categoria_interes}) por mes y año")
plt.xlabel("Mes")
plt.ylabel("Cantidad de ventas")
plt.legend(title="Año")
plt.show();

# Seleccionar las características para el modelo de árbol de decisión
X_producto = ventas_filtradas[['Mes', 'Año']]  # Mes y Año
y_producto = ventas_filtradas['transaction_qty']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_producto, y_producto, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo_arbol_producto = DecisionTreeRegressor(max_depth=4, min_samples_split=2, random_state=42)
modelo_arbol_producto.fit(X_train, y_train)

# Definir meses futuros para realizar predicciones (meses 5 a 8)
meses_futuros = pd.DataFrame({
    'Mes': [5, 6, 7, 8],  # Meses futuros
    'Año': [2024] * 4  # Años futuros
})

# Realizar predicciones para los meses futuros
ventas_futuras = modelo_arbol_producto.predict(meses_futuros)

# Mostrar las predicciones para los próximos meses
print(f"Predicciones de ventas para {producto_interes} ({categoria_interes}) en los próximos meses de 2024:")
for mes, prediccion in zip(meses_futuros['Mes'], ventas_futuras):
    print(f"Mes {mes}: {prediccion:.2f} unidades")

# Crear un DataFrame para combinar ventas reales y predicciones
meses_hist = list(ventas_resumidas.index)
ventas_hist = list(ventas_resumidas.values)
pred_df = pd.DataFrame({
    "Mes": meses_hist + list(meses_futuros['Mes']),
    "Cantidad de ventas": ventas_hist + list(ventas_futuras),
    "Tipo": ["Real"] * len(ventas_hist) + ["Predicción"] * len(ventas_futuras)
})

# Gráfico de barras con ventas reales y predicciones
plt.figure(figsize=(10, 6))
sns.barplot(data=pred_df, x="Mes", y="Cantidad de ventas", hue="Tipo", palette="Set1")
plt.title(f"Ventas reales y predicciones para {producto_interes} ({categoria_interes})")
plt.xlabel("Mes")
plt.ylabel("Cantidad de ventas")
plt.show();

# Visualizar el árbol de decisión
plt.figure(figsize=(15, 10))
plot_tree(modelo_arbol_producto, filled=True, feature_names=['Mes', 'Año'], class_names=True, rounded=True, fontsize=10)
plt.title(f"Árbol de Decisión para {producto_interes} ({categoria_interes})")
plt.show();

# Filtrar solo columnas numéricas
numeric_cols = ventas_producto.select_dtypes(include=['number'])

# Mapa de calor de correlación
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="YlGnBu")
plt.title("Matriz de correlación entre variables numéricas")
plt.show();






# Text Mining - Análisis de Sentimiento
# Cargar el dataset
data = pd.read_csv("./dataset/coffee-shop-sales-revenue.csv", delimiter='|')

# Calcular polaridad (sentimiento) de cada descripción
data['sentiment'] = data['product_detail'].apply(lambda x: TextBlob(x).sentiment.polarity)
sentiment_counts = data['sentiment'].apply(lambda x: 'Positivo' if x > 0 else ('Negativo' if x < 0 else 'Neutral')).value_counts()

# Graficar
plt.figure(figsize=(15, 4))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightgrey'])
plt.title("Distribución de sentimientos en las descripciones de productos")
plt.show()

# Calcular polaridad de cada descripción
data['polarity'] = data['product_detail'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Histograma de polaridad
plt.figure(figsize=(10, 4))
sns.histplot(data['polarity'], kde=True, bins=20)
plt.title('Distribución de la Polaridad del Sentimiento')
plt.xlabel('Polaridad')
plt.ylabel('Frecuencia')
plt.show()

# Calcular subjetividad de cada descripción
data['subjectivity'] = data['product_detail'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Histograma de subjetividad
plt.figure(figsize=(10, 4))
sns.histplot(data['subjectivity'], kde=True, bins=20, color="purple")
plt.title('Distribución de la Subjetividad del Sentimiento')
plt.xlabel('Subjetividad')
plt.ylabel('Frecuencia')
plt.show()

# Filtrar las descripciones con polaridad positiva
positive_text = ' '.join(data[data['polarity'] > 0]['product_detail'])
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

plt.figure(figsize=(15, 4))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras - Descripciones Positivas')
plt.show()

# Filtrar las descripciones con polaridad negativa
negative_text = ' '.join(data[data['polarity'] < 0]['product_detail'])
wordcloud_negative = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)

plt.figure(figsize=(15, 4))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras - Descripciones Negativas')
plt.show()
