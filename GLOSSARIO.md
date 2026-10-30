# Glossário

Os termos abaixo estão organizados por arquivo, na ordem em que aparecem no código.

# 1. Bateria/Bateria.ipynb

Já explicados antes: nenhum.

## 1.1 `numpy`
É a biblioteca usada para criar arrays numéricos e realizar operações matemáticas em matrizes, sendo a base das entradas e saídas do modelo.
Aparece em: `import numpy as np` e `np.array([...], dtype=np.float32)` em `Bateria/Bateria.ipynb`.
Exemplo: `medicoes_viagens = np.array([[0, 1], [5, 1], [5, 3], [10, 2]], dtype=np.float32)` em `Bateria/Bateria.ipynb`.
Referência: https://numpy.org/doc/stable/user/quickstart.html

## 1.2 `tensorflow`
É a biblioteca principal usada para construir e treinar o modelo de regressão linear para prever o gasto de bateria.
Aparece em: `import tensorflow as tf` em `Bateria/Bateria.ipynb`.
Exemplo: `modelo = tf.keras.Sequential([...])` em `Bateria/Bateria.ipynb`.
Referência: https://www.tensorflow.org/guide/keras/sequential_model

## 1.3 `tf.keras.Sequential`
É a classe do Keras usada para empilhar camadas em sequência e formar o modelo de regressão linear.
Aparece em: `modelo = tf.keras.Sequential([ ... ])` em `Bateria/Bateria.ipynb`.
Exemplo: `tf.keras.layers.Input(shape=(2,))` e `tf.keras.layers.Dense(1)` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/guides/sequential_model/

## 1.4 `Input`
É a camada de entrada do modelo, responsável por declarar o formato dos dados de entrada.
Aparece em: `tf.keras.layers.Input(shape=(2,))` em `Bateria/Bateria.ipynb`.
Exemplo: `Input(shape=(2,))` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/layers/core_layers/input/

## 1.5 `Dense`
É a camada totalmente conectada que produz a saída do modelo, com um neurônio neste caso.
Aparece em: `tf.keras.layers.Dense(1)` em `Bateria/Bateria.ipynb`.
Exemplo: `tf.keras.layers.Dense(1)` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/layers/core_layers/dense/

## 1.6 `compile`
É o método que configura o treinamento do modelo, define o otimizador e a função de perda.
Aparece em: `modelo.compile(optimizer=tf.keras.optimizers.Adam(0.05), loss="mse")` em `Bateria/Bateria.ipynb`.
Exemplo: `modelo.compile(optimizer=tf.keras.optimizers.Adam(0.05), loss="mse")` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/models/model_training_apis/

## 1.7 `Adam`
É o otimizador de aprendizado usado para ajustar os pesos do modelo de forma incremental.
Aparece em: `tf.keras.optimizers.Adam(0.05)` em `Bateria/Bateria.ipynb`.
Exemplo: `optimizer=tf.keras.optimizers.Adam(0.05)` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/optimizers/adam/

## 1.8 `mse`
É a função de perda Mean Squared Error, que calcula a média dos erros quadráticos do modelo.
Aparece em: `loss="mse"` em `Bateria/Bateria.ipynb`.
Exemplo: `modelo.compile(optimizer=tf.keras.optimizers.Adam(0.05), loss="mse")` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/losses/

## 1.9 `fit`
É o método que treina o modelo usando os dados de entrada e os valores esperados.
Aparece em: `modelo.fit(medicoes_viagens, gasto_viagens, epochs=1450, verbose=0)` em `Bateria/Bateria.ipynb`.
Exemplo: `epochs=1450` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/models/model_training_apis/#fit-method

## 1.10 `epochs`
É o número de passadas de treinamento sobre os dados, usado para reduzir o erro do modelo.
Aparece em: `epochs=1450` em `Bateria/Bateria.ipynb`.
Exemplo: `modelo.fit(medicoes_viagens, gasto_viagens, epochs=1450, verbose=0)` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/models/model_training_apis/#fit-method

## 1.11 `pesos`
São os coeficientes aprendidos pelo modelo, representando a influência de cada variável na previsão final.
Aparece em: `pesos, vies = modelo.layers[-1].get_weights()` e `print("a (carga):", pesos[0][0])` em `Bateria/Bateria.ipynb`.
Exemplo: `pesos[0][0]` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/layers/base_layer/#layer-weights

## 1.12 `vies`
É o termo de bias do modelo, que desloca a previsão mesmo quando as entradas não mudam.
Aparece em: `pesos, vies = modelo.layers[-1].get_weights()` e `print("c (viés):", vies[0])` em `Bateria/Bateria.ipynb`.
Exemplo: `vies[0]` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/layers/base_layer/#layer-weights

## 1.13 `gasto_previsto`
É a saída final do modelo para uma nova viagem, mostrando o gasto estimado de bateria.
Aparece em: `gasto_previsto = modelo(nova_viagem, training=False).numpy()` e `print("Gasto previsto:", gasto_previsto[0][0])` em `Bateria/Bateria.ipynb`.
Exemplo: `gasto_previsto[0][0]` em `Bateria/Bateria.ipynb`.
Referência: https://www.tensorflow.org/guide/keras/sequential_model

# 2. Prints/Prints.py

Já explicados antes: `numpy (1.1)`, `tensorflow (1.2)`, `mse (1.8)`, `pesos (1.11)`, `vies (1.12)`, `gasto_previsto (1.13)`.

## 2.1 `mostrar_resultados`
É a função que imprime no console os valores inseridos, a perda final e os parâmetros aprendidos pelo modelo.
Aparece em: `def mostrar_resultados(carga, distancia, perdas, pesos, vies, gasto_previsto):` em `Prints/Prints.py`.
Exemplo: `print("\nLoss final (MSE):", perdas)` em `Prints/Prints.py`.
Referência: https://docs.python.org/3/library/functions.html#print

## 2.2 `perdas`
É a variável que guarda a perda final do modelo, usada para indicar a qualidade da previsão.
Aparece em: `mostrar_resultados(carga, distancia, perdas, pesos, vies, gasto_previsto)` e `print("\nLoss final (MSE):", perdas)` em `Prints/Prints.py`.
Exemplo: `print("\nLoss final (MSE):", perdas)` em `Prints/Prints.py`.
Referência: https://keras.io/api/losses/

## 2.3 `print`
É a função nativa do Python para exibir mensagens no console, usada para mostrar resultados do modelo.
Aparece em: `print("\nValores inseridos: Carga:", carga, "kg, Distância:", distancia, "km")` e demais chamadas em `Prints/Prints.py`.
Exemplo: `print("\nGasto previsto %:", gasto_previsto[0][0])` em `Prints/Prints.py`.
Referência: https://docs.python.org/3/library/functions.html#print

## 2.4 `carga`
É a massa transportada na viagem, uma das duas variáveis de entrada do modelo de regressão linear.
Aparece em: `def mostrar_resultados(carga, distancia, perdas, pesos, vies, gasto_previsto)` e `print("\nValores inseridos: Carga:", carga, "kg, Distância:", distancia, "km")` em `Prints/Prints.py`.
Exemplo: `print("\nValores inseridos: Carga:", carga, "kg, Distância:", distancia, "km")` em `Prints/Prints.py`.
Referência: https://numpy.org/doc/stable/user/quickstart.html

## 2.5 `distancia`
É a distância percorrida em quilômetros, a segunda variável de entrada do modelo.
Aparece em: `def mostrar_resultados(carga, distancia, perdas, pesos, vies, gasto_previsto)` e `print("\nValores inseridos: Carga:", carga, "kg, Distância:", distancia, "km")` em `Prints/Prints.py`.
Exemplo: `print("\nValores inseridos: Carga:", carga, "kg, Distância:", distancia, "km")` em `Prints/Prints.py`.
Referência: https://numpy.org/doc/stable/user/quickstart.html

## 2.6 `RMSE`
É a raiz do erro quadrático médio, usada para interpretar a qualidade da previsão em uma escala mais amigável.
Aparece em: `# Calculando a raiz quadrada do loss final (RMSE = sqrt(0.68) ≈ 0.82), ...` em `Prints/Prints.py`.
Exemplo: `RMSE = sqrt(0.68) ≈ 0.82` em `Prints/Prints.py`.
Referência: https://en.wikipedia.org/wiki/Root-mean-square_deviation

## 2.7 `loss`
É a função de perda usada durante o treinamento para medir o quão distante a previsão ficou do valor esperado.
Aparece em: `print("\nLoss final (MSE):", perdas)` e no comentário `# O MSE ...` em `Prints/Prints.py`.
Exemplo: `loss="mse"` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/losses/

## 2.8 `float32`
É o tipo de dado em ponto flutuante de 32 bits usado para armazenar valores numéricos do modelo com precisão suficiente e baixo custo computacional.
Aparece em: `dtype=np.float32` em `Bateria/Bateria.ipynb`.
Exemplo: `medicoes_viagens = np.array([...], dtype=np.float32)` em `Bateria/Bateria.ipynb`.
Referência: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float32

## 2.9 `verbose`
É o parâmetro que controla o nível de saída do treinamento; com `0`, a execução fica silenciosa.
Aparece em: `modelo.fit(medicoes_viagens, gasto_viagens, epochs=1450, verbose=0)` em `Bateria/Bateria.ipynb`.
Exemplo: `verbose=0` em `Bateria/Bateria.ipynb`.
Referência: https://keras.io/api/models/model_training_apis/#fit-method

## 2.10 `mean squared error`
É o conceito estatístico por trás da perda MSE, medindo a média dos erros quadráticos entre previsto e real.
Aparece em: `# O MSE (Mean Squared Error) mede a média dos erros ao quadrado.` em `Prints/Prints.py`.
Exemplo: `MSE = média de (valor_previsto - valor_real)^2` em `Prints/Prints.py`.
Referência: https://en.wikipedia.org/wiki/Mean_squared_error
