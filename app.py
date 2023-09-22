import streamlit as st
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

import matplotlib.pyplot as plt

np.random.seed(0)

st.sidebar.title('Documentación')
age = st.sidebar.slider('Edad en la empresa', 0, 50, 25)

st.markdown('''
    # Reporte de ventas de zapatos
                
    Este reporte muestra las ventas de zapatos durante un año.
                
    En el reporte se identifica un crecimiento en las ventas 
    de zapatos en el mes de diciembre.

    ## Datos
                
    A continuación cargamos algunos datos y los desplegamos.

    Estos datos, indican ventas de zapatos por años trabajando en la empresa.
    Es decir, para un vendedor que lleva 5 años en la empresa, se indica cuántos zapatos
    vende en promedio por año.
''')

x = np.linspace(0, 50, 51)
y = x + 12 * np.random.random(len(x))

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
st.plotly_chart(fig)

st.markdown('''
    ## Modelado de datos
    
    En los datos se observa un comportamiento lineal, 
    por lo que se realiza un ajuste de una línea recta
    con una regresión lineal simple.
            
    $$y = m*x + b$$ 
    
            Hola
''')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1, )))
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mse')

history = model.fit(x, y, epochs=2000, verbose=0)

st.write('Pérdida final:', history.history['loss'][-1])

x_loss = np.linspace(1, len(history.history['loss']),
                     len(history.history['loss']))
y_loss = history.history['loss']

fig = plt.figure()
plt.plot(x_loss, y_loss)
st.pyplot(fig)

y_pred = model.predict(x)
y_pred = y_pred.reshape(-1)

fig_2 = go.Figure()
fig_2.add_trace(go.Scatter(x=x, y=y, mode='markers'))
fig_2.add_trace(go.Scatter(x=x, y=y_pred, mode='lines'))
st.plotly_chart(fig_2)

st.write("La edad en la empresa es", age)
expected_val = model.predict([age])
st.write("Se esperan vender", expected_val[0][0], "zapatos")