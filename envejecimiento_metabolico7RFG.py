import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np


# Configuración de la página
st.set_page_config(layout="wide", page_title="Dashboard de Envejecimiento Metabólico")

# Cargar el modelo optimizado
model = joblib.load("random_forest_optimized.pkl")
# Título del dashboard
st.title("Tablero para monitoreo y predicción de Envejecimiento Metabólico")

# Agregar estilos CSS para el fondo de cada bloque
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;  /* Color de fondo gris claro */
        padding: 20px;             /* Espaciado interno */
        border-radius: 10px;        /* Bordes redondeados */
        margin: 20px 0;             /* Espaciado entre secciones */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Sombra ligera */
    }
    </style>
""", unsafe_allow_html=True)

uploaded_file = "Final_Dataset_v.csv"

datos_diag = ["EDAD", "PESO", "TALLA", "CINTURA", "FC", "FR","SAT", "GLU", "CADERA", "BRAZO_DER","BRAZO_IZQ", "PIERNA_DER","PIERNA_IZQ","AGUA",
        "MUSCULO","GRASA","EDAD_METABOLICA","GRASA_VISCERAL","TASY","TADI","WHtR","ENV"]

nombres_columnas = {
    "EDAD": '🎂 Edad (años)',
    "PESO": '⚖️ Peso (kg)',
    "TALLA": '📏 Talla (m)',
    "CINTURA": '📐 Cintura (cm)',
    "FC": '❤️ Frec. cardiaca (lpm)',
    "FR": '🌬️ Frec. respiratoria',
    "SAT": '🩸 Saturación (%)',
    "GLU": '🔬 Glucosa (mg/dL)',
    "CADERA": '🦵 Cadera (cm)',
    "BRAZO_DER": '💪 Cir. brazo derecho (cm)',
    "BRAZO_IZQ": '💪 Cir. brazo izquierdo (cm)',
    "PIERNA_DER": '🦶 Cir. pierna derecha (cm)',
    "PIERNA_IZQ": '🦶 Cir. pierna izquierda (cm)',
    "AGUA": '💧 Agua (%)',
    "MUSCULO": '💪 Masa muscular (%)',
    "GRASA": '⚖️ Grasa (%)',
    "EDAD_METABOLICA": '🧠 Edad metabólica (años)',
    "GRASA_VISCERAL": '🩸 Grasa visceral',
    "TASY": '🩺 Tensión arterial sistólica (mmHg)',
    "TADI": '🩺 Tensión arterial diastólica (mmHg)',
    "WHtR": '⚖️ Rel. peso-talla',
    "ENV": '🔄 Envejecido (1: Sí, 0: No)',
    "SEXO": '🚻 Sexo (1: Hombre, 2: Mujer)'
}


def obtener_promedios_por_sexo(df):
    """
    Calcula el promedio de los parámetros para hombres y mujeres en el DataFrame.
    """
    promedios = df.groupby('SEXO').mean(numeric_only=True).to_dict()
    return promedios

def generar_diagnostico(promedios):
    """
    Genera un diagnóstico basado en los valores promedio de cada parámetro y devuelve un DataFrame.
    """
    diagnosticos = []
    limites = {
        "EDAD": [(40, "Adulto joven"), (60, "Adulto mayor"), (100, "Tercera edad")],
        "EDAD_METABOLICA": [(40, "Adulto joven"), (60, "Adulto mayor"), (100, "Tercera edad")],
        "PESO": [(50, "Bajo peso"), (90, "Normal"), (120, "Sobrepeso")],
        "TALLA": [(140, "Baja"), (175, "Media"), (200, "Alta")],
        "CINTURA": {1: [(94, "Riesgo aumentado"), (102, "Riesgo alto")],
                     2: [(80, "Riesgo aumentado"), (88, "Riesgo alto")]},
        "FC": [(60, "Normal"), (100, "Elevada")],
        "FR": [(12, "Normal"), (20, "Elevada")],
        "SAT": [(95, "Normal"), (90, "Baja")],
        "GLU": [(70, "Normal"), (99, "Normal"), (125, "Prediabetes"), (126, "Diabetes")],
        "CADERA": [(80, "Baja"), (100, "Normal"), (120, "Alta")],
        "BRAZO_DER": [(20, "Pequeño"), (35, "Normal"), (45, "Grande")],
        "BRAZO_IZQ": [(20, "Pequeño"), (35, "Normal"), (45, "Grande")],
        "PIERNA_DER": [(40, "Pequeño"), (60, "Normal"), (80, "Grande")],
        "PIERNA_IZQ": [(40, "Pequeño"), (60, "Normal"), (80, "Grande")],
        "IMC": [(18.5, "Bajo peso"), (24.9, "Normal"), (29.9, "Sobrepeso"), (30, "Obesidad")],
        "AGUA": {1: [(50, "Bajo"), (65, "Normal")], 2: [(45, "Bajo"), (60, "Normal")]},
        "MUSCULO": {1: [(33, "Bajo"), (50, "Normal")], 2: [(24, "Bajo"), (42, "Normal")]},
        "GRASA": {1: [(10, "Baja"), (20, "Normal"), (30, "Elevada")], 2: [(20, "Baja"), (30, "Normal"), (40, "Elevada")]},
        "EDAD_METABOLICA": [(40, "Normal"), (60, "Elevada"), (100, "Muy elevada")],
        "GRASA_VISCERAL": [(9, "Normal"), (10, "Alta")],
        "TASY": [(120, "Normal"), (129, "Elevada"), (139, "Hipertensión grado 1"), (140, "Hipertensión grado 2")],
        "TADI": [(80, "Normal"), (89, "Hipertensión grado 1"), (90, "Hipertensión grado 2")],
        "MAR": [(1.0, "Bajo"), (1.5, "Medio"), (2.0, "Alto")],
        "WHtR": [(0.5, "Normal"), (0.6, "Riesgo alto")],
        "ENV": [(0.4, "Bajo"), (0.6, "Normal"), (0.8, "Alto")],
    }
    
    for parametro, valores in promedios.items():
        if parametro not in limites:
            print(f"Parámetro {parametro} no encontrado en límites. Se omitirá.")
            continue
        
        for sexo, valor in valores.items():
            ref = limites[parametro]
            if isinstance(ref, dict):
                ref = ref.get(sexo, [])
            if ref is None:
                print(f"No hay referencia definida para {parametro} en sexo {sexo}.")
                continue
            
            estado = "Sin referencia"
            for limite, categoria in ref:
                if valor <= limite:
                    estado = categoria
                    break
            else:
                estado = "Muy alto"
            diagnosticos.append({"SEXO": sexo, "PARAMETRO": parametro, "VALOR": valor, "ESTADO": estado})
    
    df_diagnostico = pd.DataFrame(diagnosticos)
    relevantes = df_diagnostico[df_diagnostico["ESTADO"] != "Normal"]["PARAMETRO"].unique().tolist()
    
    return df_diagnostico, relevantes

def evaluar_desde_dataframe(df):
    """
    Obtiene los promedios por sexo y genera los diagnósticos correspondientes.
    """
    promedios = obtener_promedios_por_sexo(df)
    
    diagnostico_df, relevantes = generar_diagnostico(promedios)
    if diagnostico_df.empty:
        print("Advertencia: No se generaron diagnósticos. Verifica los nombres de las columnas.")
    return diagnostico_df, relevantes

def calcular_imc(peso, talla):
    if talla > 0:
        imc = peso / (talla ** 2)
        if imc < 18.5:
            categoria = "Bajo peso"
        elif 18.5 <= imc < 24.9:
            categoria = "Normal"
        elif 25 <= imc < 29.9:
            categoria = "Sobrepeso"
        else:
            categoria = "Obesidad"
        return imc, categoria
    else:
        return None, "Talla no válida"

def calcular_whtr(cintura, talla):
    if talla > 0:
        whtr = cintura / talla
        if whtr < 0.5:
            riesgo = "Bajo"
        else:
            riesgo = "Alto"
        return whtr, riesgo
    else:
        return None, "Talla no válida"

def calcular_riesgo_y_sugerencias(input_data):
    riesgos = []
    sugerencias = []
 
    cintura, glu, whtr, imc, colesterol, tasy, trig, cadera, tadi, edad, talla, fc, sexo_v, env, glu, fc = input_data
    # Evaluación de riesgos
    if whtr > 0.5:
        riesgos.append("Alto WHtR")
        sugerencias.append("Considerar un plan de alimentación saludable y ejercicio regular.")

    if cadera > 100:
        riesgos.append("Cadera amplia")
        sugerencias.append("Monitorear la distribución de la grasa corporal y mantener actividad física.")

    if imc > 25:
        riesgos.append("IMC elevado")
        sugerencias.append("Consultar con un nutricionista para planificar una dieta equilibrada.")

    if tasy > 130:
        riesgos.append("Presión arterial sistólica alta")
        sugerencias.append("Controlar la presión regularmente y reducir el consumo de sal.")

    if tadi > 85:
        riesgos.append("Presión arterial diastólica alta")
        sugerencias.append("Realizar actividad física moderada y controlar el estrés.")

    return riesgos, sugerencias

def diagnostico_sugerido_enf(input_data):
    diagnostico = []
    
    if len(input_data) < 9:
        return ["❌ Error: No hay suficientes datos para el diagnóstico."]

    cintura, glu, whtr, imc, colesterol, tasy, trig, cadera, tadi, edad, talla, fc, sexo_v, env, glu, fc = input_data

    # Asegurar que env tiene un valor válido
    if env is None:
        env = 0

    # Diagnósticos
    if imc > 25 and tasy > 130 and fc > 100 and tadi > 80:
        diagnostico.append("Disminución del gasto cardíaco, relacionado con antecedentes de enfermedades crónico-degenerativas, manifestado por IMC elevado, frecuencia cardíaca elevada e hipertensión arterial.")
            
    if glu > 100 and env > 0.5 and imc > 25:
        diagnostico.append("Riesgo de síndrome metabólico, manifestado por niveles elevados de glucosa, edad metabólica e IMC.")
        
    if env > 0.5 and imc > 25:
        diagnostico.append("Desequilibrio nutricional por ingesta superior a las necesidades metabólicas, relacionado con enfermedades crónico-degenerativas, manifestado por IMC elevado y edad metabólica aumentada.")
        
    if glu > 100:
        diagnostico.append("Riesgo de glucosa inestable, relacionado con niveles elevados de glucosa y peso corporal.")
        
    return diagnostico if diagnostico else ["✅ No se identificaron problemas en el diagnóstico de enfermería."]

def diagnostico_sugerido_med(input_data):
    diagnostico = []
    
    if len(input_data) < 9:
        return ["❌ Error: No hay suficientes datos para el diagnóstico."]

    cintura, glu, whtr, imc, colesterol, tasy, trig, cadera, tadi, edad, talla, fc, sexo_v, env, glu, fc = input_data      
    # Asegurar que env tiene un valor válido
    if env is None:
        env = 0

    # Diagnósticos
    if imc > 25 and tasy > 130 and fc > 100 and tadi > 80:
        diagnostico.append("Síndrome metabólico.")
            
    if glu > 100 and env > 0.5 and imc > 25:
        diagnostico.append("Riesgo de infarto agudo de miocardio (IAM).")

    return diagnostico if diagnostico else ["✅ No se identificaron problemas en el diagnóstico médico."]


tab1, tab2, tab3, tab4= st.tabs([
    "📋 Métricas relevantes",
    "📊 Relaciones Clínicas Importantes",
    "📈 Predicción de Envejecimiento Metabólico",
    "📖 Referencias", 
])

df = None
if uploaded_file:
    # Leer archivo
    df = pd.read_csv(uploaded_file)

    with tab1:             
        diagnosticos, relevantes = evaluar_desde_dataframe(df)
        # Crear dos contenedores visuales: uno para Mujeres y otro para Hombres
        with st.container():
            st.markdown('<div class="box">', unsafe_allow_html=True)  # Inicia el bloque con fondo

            st.markdown("## Mujeres")

            # Crear dos filas con 8 columnas cada una para Mujeres
            row1_cols = st.columns(8)
            row2_cols = st.columns(8)

            mujeres_relevantes = sorted(
            [dato for dato in relevantes if any((row["SEXO"] == 2 and row["PARAMETRO"] == dato) for _, row in diagnosticos.iterrows())]
            )
        
            # Dividir en dos partes iguales
            mitad_mujeres = len(mujeres_relevantes) // 2
            mujeres_primera_mitad = mujeres_relevantes[:mitad_mujeres]
            mujeres_segunda_mitad = mujeres_relevantes[mitad_mujeres:]
            
            # Mostrar métricas en las filas de Mujeres
            for i, dato in enumerate(mujeres_primera_mitad):
                col_index = i % 8
                with row1_cols[col_index]:
                    for _, row in diagnosticos.iterrows():
                        if row["SEXO"] == 2 and row["PARAMETRO"] == dato:
                            st.metric(label=nombres_columnas.get(row["PARAMETRO"], row["PARAMETRO"]), value=f"{row['VALOR']:.1f}", delta=row["ESTADO"])

            for i, dato in enumerate(mujeres_segunda_mitad):
                col_index = i % 8
                with row2_cols[col_index]:
                    for _, row in diagnosticos.iterrows():
                        if row["SEXO"] == 2 and row["PARAMETRO"] == dato:
                            st.metric(label=nombres_columnas.get(row["PARAMETRO"], row["PARAMETRO"]), value=f"{row['VALOR']:.1f}", delta=row["ESTADO"])

            st.markdown('</div>', unsafe_allow_html=True)  # Finaliza el bloque con fondo

        # Crear el bloque visual para Hombres
        with st.container():
            st.markdown('<div class="box">', unsafe_allow_html=True)

            st.markdown("## Hombres")

            # Crear dos filas con 8 columnas cada una para Hombres
            row3_cols = st.columns(8)
            row4_cols = st.columns(8)

            hombres_relevantes = sorted(
            [dato for dato in relevantes if any((row["SEXO"] == 1 and row["PARAMETRO"] == dato) for _, row in diagnosticos.iterrows())]
            )
    
            # Dividir en dos partes iguales
            mitad_hombres = len(hombres_relevantes) // 2
            hombres_primera_mitad = hombres_relevantes[:mitad_hombres]
            hombres_segunda_mitad = hombres_relevantes[mitad_hombres:]

            # Mostrar métricas en las filas de Hombres
            for i, dato in enumerate(hombres_primera_mitad):
                col_index = i % 8
                with row3_cols[col_index]:
                    for _, row in diagnosticos.iterrows():
                        if row["SEXO"] == 1 and row["PARAMETRO"] == dato:
                            st.metric(label=nombres_columnas.get(row["PARAMETRO"], row["PARAMETRO"]), value=f"{row['VALOR']:.1f}", delta=row["ESTADO"])

            for i, dato in enumerate(hombres_segunda_mitad):
                col_index = i % 8
                with row4_cols[col_index]:
                    for _, row in diagnosticos.iterrows():
                        if row["SEXO"] == 1 and row["PARAMETRO"] == dato:
                            st.metric(label=nombres_columnas.get(row["PARAMETRO"], row["PARAMETRO"]), value=f"{row['VALOR']:.1f}", delta=row["ESTADO"])

            st.markdown('</div>', unsafe_allow_html=True)


    with tab2:        
        selected_col = st.selectbox("Selecciona una variable para el histograma", df.columns)
        col1, col2 = st.columns(2)
        with col1:
            
            st.write(f"### Distribución de la variable: {selected_col}")
        
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data=df, x=selected_col, bins=20, kde=True, ax=ax)
            ax.grid(True, linestyle="--", alpha=0.7)
            st.pyplot(fig)

        with col2:
            st.write("### Comparación por sexo (1: Hombre, 2: Mujer)")
            if 'SEXO' in df.columns:
                #compare_var = st.selectbox("Selecciona la variable a comparar", df.columns)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x='SEXO', y=selected_col, data=df, ax=ax)
                ax.grid(True, linestyle="--", alpha=0.7)
                st.pyplot(fig)
        
        st.write(f"### Comparación entre variables:")
        col1, col2 = st.columns(2)          
           
        with col1:
            x_var = st.selectbox("Selecciona la variable X", df.columns)
            
        with col2:
            y_var = st.selectbox("Selecciona la variable Y", df.columns) 
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)

    with tab3:
        st.markdown("### ✏️ Ingrese los valores de las variables predictoras")

        # Entradas para IMC y WHtR
        st.markdown("#### 📏 Datos Antropométricos")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            peso = st.number_input("⚖️ Peso (Kg)", min_value=10.0, max_value=300.0, step=0.1, format="%.2f")
        with col2:
            talla = st.number_input("📏 Altura (cm)", min_value=140.0, max_value=200.0, step=0.1, format="%.2f")
        with col3:
            cintura = st.number_input("📐 Circunferencia cintura (cm)", min_value=50.0, max_value=150.0, step=0.1, format="%.2f")
        with col4:
            cadera = st.number_input("🦵 Circunferencia cadera (cm)", min_value=70.0, max_value=140.0, step=0.1, format="%.2f")
        with col5:
            tasy = st.number_input("🩺 Presión sistólica (mmHg)", min_value=80.0, max_value=200.0, step=1.0, format="%.2f")
        with col6:
            tadi = st.number_input("🩺 Presión diastólica (mmHg)", min_value=50.0, max_value=120.0, step=1.0, format="%.2f")

        # 🧮 **Cálculo de IMC y WHtR**
        col1, col2 = st.columns(2)
        if talla > 0.0 and peso > 0.0 and cintura > 0.0:
            imc = peso / ((talla / 100) ** 2)  # Convertir talla a metros
            whtr = cintura / talla  # No se multiplica por 100

            with col1:
                st.write(f"🟢 **IMC Calculado:** {imc:.2f}")
            with col2:
                st.write(f"🔵 **Rel. Cintura-Talla:** {whtr:.2f}")
        else:
            st.warning("⚠️ Ingrese valores válidos para Peso, Altura y Cintura para calcular IMC y WHtR.")

        # 🔬 **Datos Clínicos**
        st.markdown("#### 🔬 Datos Clínicos")
        col1, col2, col3, col4, col5, col6 = st.columns(6)  

        with col1:
            sexo = st.selectbox("🚻 Género", ["Masculino", "Femenino"])
        with col2:
            glu = st.number_input("🔬 Glucosa en sangre (mg/dL)", min_value=50.0, max_value=300.0, step=0.1, format="%.2f")
        with col3:
            fc = st.number_input("❤️ Frecuencia cardiaca (bpm)", min_value=40.0, max_value=120.0, step=1.0, format="%.2f")
        with col4:
            edad = st.number_input("🎂 Edad (años)", min_value=18, max_value=100, step=1)
        with col5:
            colesterol = st.number_input("🩸 Colesterol HDL (mg/dL)", min_value=20.0, max_value=300.0, step=0.1, format="%.2f")
        with col6:
            trig = st.number_input("⚖️ Triglicéridos (mg/dL)", min_value=50.0, max_value=500.0, step=1.0, format="%.2f")
        
        # Convertir el género a valor numérico
        sexo_v = 1 if sexo == "Masculino" else 0
        
        print("Clases del modelo:", model.classes_)                    
        
        if st.button("📊 Predecir"):                    
            input_data = np.array([[cintura, glu, whtr, imc, colesterol, tasy, trig, cadera, tadi, edad, talla, fc, sexo_v]])

            try:
                # Verificar si los valores calculados son válidos
                if whtr is None or imc is None:
                    st.error("⚠️ Error en el cálculo de IMC o WHtR. Verifique los valores ingresados.")
                else:
                    
                                     
                    # **Corregir la predicción asegurando entrada 2D**
                    prediction = model.predict(input_data)  # 🔥 Solución

                    probabilities = model.predict_proba(input_data)   
                    # Mostrar el resultado directamente
                    st.write(f"### 🔮 Resultado de la Predicción: **{prediction[0]}**")
                    
                    # Crear un diccionario con las clases y sus probabilidades
                    class_probabilities = dict(zip(model.classes_, probabilities[0]))

                    # Mostrar las probabilidades en Streamlit
                    st.write("### 🔢 Probabilidades de Predicción:")
                    for clase, prob in class_probabilities.items():
                        st.write(f"- **{clase}**: {prob:.2%}")
                                             
                    # Guardar la probabilidad correspondiente en la variable `pred`
                    pred_index = list(model.classes_).index(prediction)  # Encuentra el índice de la clase predicha
                    pred = probabilities[0][pred_index]  # Obtiene la probabilidad de esa clase
                                        # Guardar valores para diagnóstico
                    # Convertir input_data en lista
                    input_data_list = input_data.tolist()[0]  # Convierte de ndarray a lista

                    # Agregar valores adicionales
                    input_data_list.append(pred)
                    input_data_list.append(glu)
                    input_data_list.append(fc)

                    # Convertir de nuevo a ndarray si es necesario
                    input_data = np.array([input_data_list])

                    # Diagnóstico basado en valores adicionales
                    st.write("Diagnóstico")
                        
                    diage = diagnostico_sugerido_enf(input_data[0])
                    diagm = diagnostico_sugerido_med(input_data[0])

                    st.markdown("### Diagnósticos sugeridos")
                    if diage:
                        st.markdown("##### 🏥 Enfermería:")
                        for d in diage:
                            st.write(f"- {d}")
                    else:
                        st.success("✅ No se identificaron riesgos en el diagnóstico de enfermería.")

                    if diagm:
                        st.markdown("##### 🩺 Médico:")
                        for d in diagm:
                            st.write(f"- {d}")
                    else:
                        st.success("✅ No se identificaron riesgos en el diagnóstico médico.")
                        
                    # Análisis de riesgo y sugerencias
                    riesgos, sugerencias = calcular_riesgo_y_sugerencias(input_data[0])
                    if riesgos:
                        st.markdown("### Riesgos detectados y sugerencias:")
                        st.write("Los siguientes factores indican un mayor riesgo:")
                        for riesgo in riesgos:
                            st.write(f"- {riesgo}")

                        #st.markdown("### Sugerencias Personalizadas")
                        st.write("Basado en sus parámetros, se recomiendan las siguientes acciones:")
                        for sugerencia in sugerencias:
                            st.write(f"- {sugerencia}")
                    else:
                        st.success("No se identificaron riesgos significativos. Mantenga su estilo de vida saludable.")

            except Exception as e:
                st.error(f"🚨 Ha ocurrido un error en la predicción: {e}")

    with tab4: 
        st.write("Referencias.")



else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
