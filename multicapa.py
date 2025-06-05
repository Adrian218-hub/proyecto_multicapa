import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber 
from scikeras.wrappers import KerasRegressor, KerasClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(page_title="Diamond Price Prediction", layout="wide", page_icon="💎")
st.title("💎 Sistema Avanzado de Predicción de Precios de Diamantes")

# Cargar y preparar datos con caché
@st.cache_data
def load_and_prepare_data():
    try:
        data = pd.read_csv('diamonds.csv')
        st.info("Datos cargados desde diamonds.csv local")
    except Exception:
        from sklearn.datasets import fetch_openml
        diamonds = fetch_openml(name='diamonds', version=1, as_frame=True)
        data = diamonds.frame
        st.info("Datos cargados desde OpenML")
    
    data = data.dropna()
    
    # Codificación avanzada de variables categóricas
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    
    if 'cut' in data.columns:
        data['cut'] = pd.Categorical(data['cut'], categories=cut_order, ordered=True).codes
    if 'color' in data.columns:
        data['color'] = pd.Categorical(data['color'], categories=color_order, ordered=True).codes
    if 'clarity' in data.columns:
        data['clarity'] = pd.Categorical(data['clarity'], categories=clarity_order, ordered=True).codes
    
    # Selección de características con análisis de importancia
    features = ['carat', 'x', 'y', 'z', 'cut', 'color', 'clarity']
    features = [f for f in features if f in data.columns]
    
    X = data[features]
    y_reg = data['price']
    
    # Clasificación con rangos más significativos
    price_bins = [0, 3000, 10000, np.inf]
    price_labels = ['Económico', 'Intermedio', 'Premium']
    y_clas = pd.cut(data['price'], bins=price_bins, labels=price_labels)
    
    # Escalado robusto
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # División estratificada para clasificación
    X_train, X_test, y_train_reg, y_test_reg, y_train_clas, y_test_clas = train_test_split(
        X_scaled, y_reg, y_clas, test_size=0.2, random_state=42, stratify=y_clas)
    
    return X_train, X_test, y_train_reg, y_test_reg, y_train_clas, y_test_clas, scaler, features, data

X_train, X_test, y_train_reg, y_test_reg, y_train_clas, y_test_clas, scaler, features, data = load_and_prepare_data()

# Visualización exploratoria de datos
def show_data_exploration(data):
    st.header("🔍 Análisis Exploratorio de Datos")
    
    tab1, tab2, tab3 = st.tabs(["Distribuciones", "Correlaciones", "Precios"])
    
    with tab1:
        fig = make_subplots(rows=2, cols=2)
        fig.add_trace(go.Histogram(x=data['carat'], name='Carat'), row=1, col=1)
        fig.add_trace(go.Histogram(x=data['x'], name='X'), row=1, col=2)
        fig.add_trace(go.Histogram(x=data['y'], name='Y'), row=2, col=1)
        fig.add_trace(go.Histogram(x=data['z'], name='Z'), row=2, col=2)
        fig.update_layout(height=600, title_text="Distribución de Características")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        numeric_data = data.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(title="Matriz de Correlación", height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.box(data, x='cut', y='price', color='color', 
                    title="Distribución de Precios por Corte y Color")
        st.plotly_chart(fig, use_container_width=True)

show_data_exploration(data)

# Modelos avanzados
def build_advanced_regression_model(optimizer='adam', lr=0.001, dropout_rate=0.2):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Selección del optimizador con tasa de aprendizaje
    if optimizer == 'adam':
        opt = Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=lr, momentum=0.9)
    else:
        opt = RMSprop(learning_rate=lr)
    
    # Compilación del modelo con pérdida Huber correctamente definida
    model.compile(
        optimizer=opt,
        loss=Huber(),  # CORRECCIÓN: usar la clase Huber() en lugar de 'huber_loss'
        metrics=['mae', 'mse']
    )
    
    return model


def build_advanced_classification_model(optimizer='adam', lr=0.001, dropout_rate=0.2):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=lr, momentum=0.9)
    else:
        opt = RMSprop(learning_rate=lr)
        
    model.compile(optimizer=opt, 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

# Callbacks para mejor entrenamiento
def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]

# Entrenamiento y evaluación mejorados
def train_and_evaluate_advanced(model, X_train, y_train, X_test, y_test, epochs=100, is_regression=True):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        verbose=0,
        callbacks=get_callbacks()
    )
    
    if is_regression:
        y_pred = model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
        
        # Gráfico de residuos
        residuals = y_test - y_pred
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuos'
        ))
        fig_res.add_hline(y=0, line_dash='dash', line_color='red')
        fig_res.update_layout(
            title='Análisis de Residuos',
            xaxis_title='Predicciones',
            yaxis_title='Residuos'
        )
        
        return history, mse, mae, r2, y_pred, fig_res
    else:
        y_pred_prob = model.predict(X_test)
        y_pred = y_pred_prob.argmax(axis=1)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cl_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Gráfico de matriz de confusión
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=['Económico', 'Intermedio', 'Premium'],
            y=['Económico', 'Intermedio', 'Premium'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            showscale=False
        ))
        fig_cm.update_layout(
            title='Matriz de Confusión',
            xaxis_title='Predicho',
            yaxis_title='Real'
        )
        
        # Gráfico de curvas ROC
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fig_roc = go.Figure()
        for i in range(n_classes):
            fig_roc.add_trace(go.Scatter(
                x=fpr[i],
                y=tpr[i],
                name=f'Clase {i} (AUC = {roc_auc[i]:.2f})'
            ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line=dict(dash='dash'),
            name='Aleatorio'
        ))
        
        fig_roc.update_layout(
            title='Curvas ROC',
            xaxis_title='Tasa de Falsos Positivos',
            yaxis_title='Tasa de Verdaderos Positivos'
        )
        
        return history, acc, y_pred, cl_report, fig_cm, fig_roc

# Modelos alternativos (ML clásico)
def train_ml_model(model_type, X_train, y_train, X_test, y_test, is_regression=True):
    if is_regression:
        if model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'SVR':
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        else:  # XGBoost
            model = XGBRegressor(objective='reg:squarederror', random_state=42)
    else:
        if model_type == 'Random Forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'SVC':
            model = SVC(kernel='rbf', C=1.0, probability=True)
        else:  # XGBoost
            model = XGBClassifier(objective='multi:softprob', random_state=42)
    
    model.fit(X_train, y_train)
    
    if is_regression:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = model.score(X_test, y_test)
        return mse, mae, r2, y_pred
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return acc, y_pred, cm, y_proba

# Interfaz de usuario mejorada
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    model_family = st.selectbox("Familia de Modelo", 
                              ["Red Neuronal", "Machine Learning Clásico"])
    
    if model_family == "Red Neuronal":
        model_type = st.selectbox("Tipo de modelo", ['Regresión', 'Clasificación'])
        optimizer = st.selectbox("Optimizador", ['adam', 'sgd', 'rmsprop'])
        lr = st.slider("Tasa de aprendizaje", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)
        epochs = st.slider("Épocas máximas", 10, 500, 100)
    else:
        model_type = st.selectbox("Tipo de modelo", ['Regresión', 'Clasificación'])
        ml_model = st.selectbox("Modelo ML", ['Random Forest', 'SVM', 'XGBoost'])
    
    st.header("🔮 Datos para Predicción")
    input_vals = {}
    for f in features:
        if f == 'carat':
            input_vals[f] = st.slider(f, 
                                    float(data[f].min()), 
                                    float(data[f].max()), 
                                    float(data[f].median()),
                                    step=0.01)
        elif f in ['x', 'y', 'z']:
            input_vals[f] = st.slider(f"{f} (mm)", 
                                    float(data[f].min()), 
                                    float(data[f].max()), 
                                    float(data[f].median()),
                                    step=0.1)
        elif f == 'cut':
            input_vals[f] = st.select_slider("Calidad de Corte", 
                                           options=[0, 1, 2, 3, 4],
                                           value=3,
                                           format_func=lambda x: ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'][x])
        elif f == 'color':
            input_vals[f] = st.select_slider("Color", 
                                          options=[0, 1, 2, 3, 4, 5, 6],
                                          value=3,
                                          format_func=lambda x: ['J', 'I', 'H', 'G', 'F', 'E', 'D'][x])
        elif f == 'clarity':
            input_vals[f] = st.select_slider("Claridad", 
                                          options=[0, 1, 2, 3, 4, 5, 6, 7],
                                          value=4,
                                          format_func=lambda x: ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'][x])

# Entrenamiento y visualización
if st.sidebar.button("🚀 Entrenar Modelo"):
    st.header("📊 Resultados del Modelo")
    
    if model_family == "Red Neuronal":
        with st.spinner("Entrenando red neuronal..."):
            if model_type == 'Regresión':
                model = build_advanced_regression_model(optimizer, lr, dropout_rate)
                history, mse, mae, r2, y_pred, fig_res = train_and_evaluate_advanced(
                    model, X_train, y_train_reg, X_test, y_test_reg, epochs, True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:,.2f}")
                col2.metric("MAE", f"{mae:,.2f}")
                col3.metric("R²", f"{r2:.3f}")
                
                # Gráficos de entrenamiento
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history.history['loss'],
                    name='Entrenamiento'
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    name='Validación'
                ))
                fig_loss.update_layout(
                    title='Pérdida durante el Entrenamiento',
                    xaxis_title='Época',
                    yaxis_title='Pérdida'
                )
                
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Scatter(
                    y=history.history['mae'],
                    name='Entrenamiento'
                ))
                fig_mae.add_trace(go.Scatter(
                    y=history.history['val_mae'],
                    name='Validación'
                ))
                fig_mae.update_layout(
                    title='MAE durante el Entrenamiento',
                    xaxis_title='Época',
                    yaxis_title='MAE'
                )
                
                st.plotly_chart(fig_loss, use_container_width=True)
                st.plotly_chart(fig_mae, use_container_width=True)
                st.plotly_chart(fig_res, use_container_width=True)
                
            else:  # Clasificación
                model = build_advanced_classification_model(optimizer, lr, dropout_rate)
                history, acc, y_pred, cl_report, fig_cm, fig_roc = train_and_evaluate_advanced(
                    model, X_train, y_train_clas.cat.codes, X_test, y_test_clas.cat.codes, epochs, False)
                
                st.metric("Accuracy", f"{acc:.2%}")
                
                # Mostrar reporte de clasificación
                st.subheader("Reporte de Clasificación")
                st.dataframe(pd.DataFrame(cl_report).transpose())
                
                # Gráficos de entrenamiento
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history.history['loss'],
                    name='Entrenamiento'
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    name='Validación'
                ))
                fig_loss.update_layout(
                    title='Pérdida durante el Entrenamiento',
                    xaxis_title='Época',
                    yaxis_title='Pérdida'
                )
                
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    y=history.history['accuracy'],
                    name='Entrenamiento'
                ))
                fig_acc.add_trace(go.Scatter(
                    y=history.history['val_accuracy'],
                    name='Validación'
                ))
                fig_acc.update_layout(
                    title='Accuracy durante el Entrenamiento',
                    xaxis_title='Época',
                    yaxis_title='Accuracy'
                )
                
                st.plotly_chart(fig_loss, use_container_width=True)
                st.plotly_chart(fig_acc, use_container_width=True)
                st.plotly_chart(fig_cm, use_container_width=True)
                st.plotly_chart(fig_roc, use_container_width=True)
    else:  # Machine Learning Clásico
        with st.spinner(f"Entrenando modelo {ml_model}..."):
            is_regression = model_type == 'Regresión'
            
            if is_regression:
                mse, mae, r2, y_pred = train_ml_model(
                    ml_model, X_train, y_train_reg, X_test, y_test_reg, True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:,.2f}")
                col2.metric("MAE", f"{mae:,.2f}")
                col3.metric("R²", f"{r2:.3f}")
                
                # Gráfico de predicciones vs reales
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test_reg,
                    y=y_pred,
                    mode='markers',
                    name='Predicciones'
                ))
                fig.add_trace(go.Scatter(
                    x=[y_test_reg.min(), y_test_reg.max()],
                    y=[y_test_reg.min(), y_test_reg.max()],
                    mode='lines',
                    name='Línea perfecta',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title='Predicciones vs Valores Reales',
                    xaxis_title='Valor Real',
                    yaxis_title='Predicción'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # Clasificación
                acc, y_pred, cm, y_proba = train_ml_model(
                    ml_model, X_train, y_train_clas.cat.codes, X_test, y_test_clas.cat.codes, False)
                
                st.metric("Accuracy", f"{acc:.2%}")
                
                # Matriz de confusión
                fig_cm = go.Figure(go.Heatmap(
                    z=cm,
                    x=['Económico', 'Intermedio', 'Premium'],
                    y=['Económico', 'Intermedio', 'Premium'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    showscale=False
                ))
                fig_cm.update_layout(
                    title='Matriz de Confusión',
                    xaxis_title='Predicho',
                    yaxis_title='Real'
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Curvas ROC
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_curve, auc
                
                y_test_bin = label_binarize(y_test_clas.cat.codes, classes=[0, 1, 2])
                n_classes = y_test_bin.shape[1]
                
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                fig_roc = go.Figure()
                for i in range(n_classes):
                    fig_roc.add_trace(go.Scatter(
                        x=fpr[i],
                        y=tpr[i],
                        name=f'Clase {i} (AUC = {roc_auc[i]:.2f})'
                    ))
                
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    line=dict(dash='dash'),
                    name='Aleatorio'
                ))
                
                fig_roc.update_layout(
                    title='Curvas ROC',
                    xaxis_title='Tasa de Falsos Positivos',
                    yaxis_title='Tasa de Verdaderos Positivos'
                )
                st.plotly_chart(fig_roc, use_container_width=True)

# Predicción personalizada
st.header("🔮 Predicción Personalizada")

input_df = pd.DataFrame([input_vals])
input_scaled = scaler.transform(input_df)

if st.button("✨ Predecir"):
    if 'model' not in locals():
        st.warning("Por favor, entrena un modelo primero.")
    else:
        if model_type == 'Regresión' or (model_family == "Machine Learning Clásico" and model_type == 'Regresión'):
            if model_family == "Red Neuronal":
                pred = model.predict(input_scaled)[0][0]
            else:
                pred = model.predict(input_scaled)[0]
            
            st.success(f"**Precio estimado:** ${pred:,.2f}")
            
            # Mostrar interpretación
            st.subheader("Interpretación")
            st.write(f"- **Carat:** {input_vals['carat']} quilates")
            st.write(f"- **Dimensiones:** {input_vals['x']} x {input_vals['y']} x {input_vals['z']} mm")
            st.write(f"- **Calidad de Corte:** {['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'][input_vals['cut']]}")
            st.write(f"- **Color:** {['J', 'I', 'H', 'G', 'F', 'E', 'D'][input_vals['color']]}")
            st.write(f"- **Claridad:** {['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'][input_vals['clarity']]}")
            
        else:  # Clasificación
            if model_family == "Red Neuronal":
                pred_proba = model.predict(input_scaled)[0]
            else:
                pred_proba = model.predict_proba(input_scaled)[0]
            
            pred_class = np.argmax(pred_proba)
            categories = ['Económico', 'Intermedio', 'Premium']
            
            st.success(f"**Categoría estimada:** {categories[pred_class]}")
            
            # Gráfico de probabilidades
            fig_proba = go.Figure(go.Bar(
                x=categories,
                y=pred_proba,
                text=[f"{p:.1%}" for p in pred_proba],
                textposition='auto',
                marker_color=['#636EFA', '#EF553B', '#00CC96']
            ))
            fig_proba.update_layout(
                title='Probabilidades de Categoría',
                yaxis_title='Probabilidad',
                yaxis_tickformat='.0%'
            )
            st.plotly_chart(fig_proba, use_container_width=True)
            
            # Recomendaciones basadas en la categoría
            st.subheader("Recomendaciones")
            if pred_class == 0:  # Económico
                st.info("""
                - **Mercado objetivo:** Compradores con presupuesto limitado
                - **Uso recomendado:** Joyería casual o anillos de compromiso sencillos
                - **Alternativas:** Considera diamantes ligeramente más pequeños con mejor corte
                """)
            elif pred_class == 1:  # Intermedio
                st.info("""
                - **Mercado objetivo:** Compradores que buscan equilibrio entre calidad y precio
                - **Uso recomendado:** Anillos de compromiso o joyería fina
                - **Alternativas:** Mejorar el color en 1 grado podría aumentar significativamente el valor
                """)
            else:  # Premium
                st.info("""
                - **Mercado objetivo:** Compradores de lujo y coleccionistas
                - **Uso recomendado:** Joyería de alta gama e inversión
                - **Certificación:** Considera obtener certificación GIA para maximizar el valor
                """)

# Explicación del modelo
st.header("📚 Explicación del Modelo")
with st.expander("¿Cómo funciona este sistema?"):
    st.markdown("""
    Este sistema predictivo utiliza avanzadas técnicas de aprendizaje automático para estimar:
    
    1. **Precio de diamantes** (modelo de regresión)
    2. **Categoría de precio** (modelo de clasificación)
    
    ### Características utilizadas:
    - **Carat (Quilates):** Peso del diamante (1 quilate = 0.2 gramos)
    - **Dimensiones (x, y, z):** Medidas en milímetros
    - **Cut (Corte):** Calidad del corte (Fair, Good, Very Good, Premium, Ideal)
    - **Color:** Grado de color (D el mejor, J el peor en este dataset)
    - **Clarity (Claridad):** Pureza del diamante (IF el mejor, I1 el peor)
    
    ### Metodología:
    1. **Preprocesamiento:** Normalización de datos y codificación de categorías
    2. **Modelado:** 
       - Redes Neuronales con regularización y dropout
       - Modelos clásicos como Random Forest y XGBoost
    3. **Evaluación:** Métricas robustas y visualizaciones interactivas
    
    ### Interpretación de resultados:
    - **R² (Regresión):** Proporción de varianza explicada (1 es perfecto)
    - **Accuracy (Clasificación):** Porcentaje de predicciones correctas
    - **Matriz de Confusión:** Desglose de aciertos/errores por categoría
    """)

# Notas finales
st.markdown("---")
st.caption("""
**Nota:** Los modelos de machine learning son herramientas predictivas, no deterministas. 
Los resultados deben considerarse como estimaciones basadas en patrones históricos.
""")