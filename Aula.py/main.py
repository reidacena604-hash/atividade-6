
import pandas as pd
from sklearn.linear_model import LinearRegression

# Dados
gamer = pd.DataFrame({
    'horas_jogo': [1, 2, 4, 6, 8, 10],
    'cansaco': [1, 2, 3, 5, 8, 10]
})

# Variáveis
X = gamer[['horas_jogo']]
y = gamer['cansaco']

# Modelo
model = LinearRegression()
model.fit(X, y)

# Entrada do usuário
horas = float(input("Horas jogadas: "))

# Previsão
entrada = pd.DataFrame({
    'horas_jogo': [horas]
})

predicao = model.predict(entrada)[0]

# Resultado
print(f"Cansaço estimado: {predicao:.2f}/10")

#################################################

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Configuração da página
st.set_page_config(page_title="🍦 IA do Sorvete")

# Título
st.title("🍦 IA do Sorvete")

st.write("Previsão da quantidade de sorvetes vendidos com base na temperatura.")

# Dados
sorvete = pd.DataFrame({
    'temperatura': [18, 20, 24, 27, 30, 35],
    'vendas': [20, 25, 40, 55, 70, 100]
})

# Variáveis
X = sorvete[['temperatura']]
y = sorvete['vendas']

# Modelo
model = LinearRegression()
model.fit(X, y)

# Entrada do usuário
temperatura = st.slider(
    "Temperatura (°C)",
    min_value=0,
    max_value=45,
    value=25
)

# Previsão
entrada = pd.DataFrame({
    'temperatura': [temperatura]
})

previsao = model.predict(entrada)[0]

# Resultado
st.metric(
    label="🍦 Sorvetes vendidos",
    value=f"{previsao:.0f}"
)

# Mensagens
if previsao > 80:
    st.success("🔥 Dia perfeito para vender muito sorvete!")
elif previsao > 50:
    st.info("☀️ Movimento moderado.")
else:
    st.warning("❄️ Poucas vendas previstas.")

# Mostrar dados
st.subheader("📊 Dados Utilizados")
st.dataframe(sorvete)