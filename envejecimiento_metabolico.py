import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Configuración de la página
st.set_page_config(layout="wide", page_title="Dashboard de Envejecimiento Metabólico")

# Título del dashboard
st.title("Dashboard de Envejecimiento Metabólico")

# Carga de datos
st.sidebar.title("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

# Variables para el modelo predictivo
selected_features = ['WHtR', 'CADERA', 'IMC', 'TASY', 'TADI','SEXO']
df = None
if uploaded_file:
    # Leer archivo
    df = pd.read_csv(uploaded_file)

    # Diseño de KPI's
    st.markdown("## Métricas Clave")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'EDAD' in df.columns:
            st.metric(label="Edad Promedio", value=int(df["EDAD"].mean()), delta=int(df["EDAD"].mean()) - 40)

    with col2:
        if 'EDAD_METABOLICA' in df.columns:
            st.metric(label="Edad Metabólica Promedio", value=int(df["EDAD_METABOLICA"].mean()), delta=int(df["EDAD_METABOLICA"].mean()) - 50)

    with col3:
        if 'BALANCE' in df.columns:
            st.metric(label="Balance Promedio", value=f"${df['BALANCE'].mean():,.2f}", delta=-500)

    # Análisis de datos
    st.markdown("## Análisis de Datos")
    col1, col2 = st.columns(2)

    with col1:
        x_var = st.selectbox("Selecciona la variable X", df.columns)
        y_var = st.selectbox("Selecciona la variable Y", df.columns)
        st.write(f"### Relación entre {x_var} y {y_var}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)

    with col2:
        selected_col = st.selectbox("Selecciona una variable para el histograma", df.columns)
        st.write(f"### Distribución de la variable: {selected_col}")
     
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df, x=selected_col, bins=20, kde=True, ax=ax)
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)

    # Filtro de edad
    if 'EDAD' in df.columns:
        st.sidebar.title("Filtros por Edad")
        edad_min = st.sidebar.slider("Edad mínima", int(df['EDAD'].min()), int(df['EDAD'].max()))
        edad_max = st.sidebar.slider("Edad máxima", int(df['EDAD'].min()), int(df['EDAD'].max()))
        filtered_df = df[(df['EDAD'] >= edad_min) & (df['EDAD'] <= edad_max)]

        st.markdown("## Datos Filtrados por Edad")
        st.dataframe(filtered_df)

    # Modelo predictivo
    st.sidebar.title("Predicción de envejecimiento metabólico")
    missing_columns = [col for col in selected_features if col not in df.columns]
    if missing_columns:
        st.error(f"El dataset no contiene las columnas requeridas: {missing_columns}")
    else:
        X = df[selected_features]
        y = df["ENV"] if "ENV" in df.columns else None
        
        if y is not None:
            # Preprocesamiento para manejar NaN
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)
            y = y.fillna(y.mean())

            # Entrenar el modelo
            model = LinearRegression()
            model.fit(X, y)

            st.sidebar.subheader("Ingrese los valores de las variables predictoras")
            input_data = []

            for feature in selected_features:
                value = st.sidebar.number_input(f"Ingrese el valor para {feature}", min_value=0.0, format="%.2f")
                input_data.append(value)

            if st.sidebar.button("Predecir"):
                try:
                    input_array = imputer.transform([input_data])
                    prediction = model.predict(input_array)[0]
                    result = "MUY PROBABLE" if abs(prediction - 1) < abs(prediction - 0) else "POCO PROBABLE"
                    st.write(f"### Predicción de Envejecimiento Metabólico: ({prediction:.2f}) {result}")
                    #st.write(f"### Predicción de Envejecimiento Metabólico: {prediction:.2f}")
                except Exception as e:
                    st.error(f"Ha ocurrido un error en la predicción: {e}") 
                    
        else:
            st.warning("La columna 'Edad Metabólica' no está disponible en el dataset.")


else:
    st.info("Por favor, sube un archivo CSV para comenzar.")


