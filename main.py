import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend.frequent_patterns
import mlxtend.preprocessing
import numpy
import warnings

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

