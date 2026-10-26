## 📌 O que o projeto faz
Este projeto implementa um modelo de aprendizado de máquina (Deep Learning básico com Keras) estruturado para prever o consumo percentual de uma bateria. O sistema aprende a partir de dados medidos anteriormente (relacionando peso/carga e distância) para estimar o gasto energético de forma automatizada.

---

## 💻 Ambiente de Desenvolvimento
- **Fase Inicial:** Desenvolvido e testado no ambiente em nuvem do **Google Colab**, aproveitando os recursos de execução em Python.
- **Fase Final / Organização:** Finalizado localmente no **VS Code**, utilizando a extensão do Colab para integração contínua e execução do notebook (`.ipynb`).

---

## 🚀 Como Usar
1. Clone o repositório em sua máquina ou abra o arquivo do notebook (`.ipynb`) no Google Colab ou VS Code.
2. Certifique-se de ter as bibliotecas necessárias instaladas (`tensorflow`, `numpy`).
3. Execute as células sequencialmente para carregar os dados de treino, compilar o modelo e realizar as previsões de teste.

---

## 🖼️ Imagens do Projeto
Resultados gráficos e visualizações do treinamento do modelo:

![Resultado do Treino](imagens/imagem.jpg)

---

## 🧠 O que aprendi durante o desenvolvimento do projeto

1. **Otimização do Método de Inferência:**
   Entendimento sobre a diferença de desempenho entre utilizar o método `model.predict()` e a chamada direta do objeto do modelo (`model(data, training=False).numpy()`). A chamada direta evita o *overhead* e os avisos de *retracing* do TensorFlow (`tf.function`) em execuções de inferência individual, resultando em um código mais limpo e execução mais eficiente.

2. **Calibração Empírica de Hiperparâmetros (Ajuste de Épocas):**
   Compreensão sobre o impacto da quantidade de épocas no tempo de execução e na convergência do modelo. O material de estudo sugeria 3.000 épocas, porém, após testes empíricos controlados (avaliando o comportamento a cada variação de épocas), identificou-se o valor de **1.450 épocas** como a quantidade ideal para alcançar os parâmetros matemáticos desejados ($a \approx 0.99$, $b \approx 9.11$, $c \approx -1.89$) mantendo o tempo de execução reduzido para cerca de 1 minuto.

---

## 🛠️ Stack Tecnológica e Referências

Para manter o código-fonte limpo, direto e sem poluição de comentários excessivos, as bibliotecas, classes, métodos e hiperparâmetros utilizados no script são documentados abaixo com suas respectivas referências oficiais:

### 1. Manipulação de Dados (`NumPy`)
- **`np.array`**: Cria tabelas estruturadas de dados (arrays multidimensionais) para processamento no TensorFlow. ([Documentação numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html))
- **`dtype=np.float32`**: Define o tipo de dado como decimal de 32 bits, padrão utilizado em modelos de precisão simples no TensorFlow. ([Documentação Tipos NumPy](https://numpy.org/doc/stable/user/basics.types.html))

### 2. Construção da Rede Neural (`TensorFlow / Keras`)
- **`tf.keras.Sequential`**: Agrupa uma pilha linear de camadas neurais para o fluxo sequencial de dados. ([Documentação tf.keras.Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential))
- **`tf.keras.layers.Input`**: Define o formato de entrada (*shape*) dos dados na rede (2 variáveis: carga e distância). ([Documentação tf.keras.layers.Input](https://www.tensorflow.org/api_docs/python/tf/keras/Input))
- **`tf.keras.layers.Dense`**: Camada densamente conectada que realiza a operação linear $a \times \text{carga} + b \times \text{distância} + c$. ([Documentação tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense))

### 3. Otimização e Treinamento
- **`compile()`**: Configura o modelo definindo o otimizador e a função de custo. ([Documentação Model.compile](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile))
- **`optimizer=tf.keras.optimizers.Adam`**: Otimizador adaptativo que ajusta os pesos e o viés a cada passo do algoritmo. ([Documentação tf.keras.optimizers.Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam))
- **`loss="mse"` (Mean Squared Error)**: Função de perda que calcula a média dos quadrados dos erros para orientar o ajuste dos pesos. ([Documentação MeanSquaredError](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError))
- **`fit()`**: Executa o ciclo de treinamento com os dados de entrada e saída. ([Documentação Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit))
- **`epochs`**: Hiperparâmetro que define a quantidade de ciclos completos de treinamento sobre o conjunto de dados (otimizado para **1.450 épocas**).
- **`verbose=0`**: Oculta a exibição do log de progresso de cada época no terminal/notebook durante o treinamento.

### 4. Extração de Parâmetros e Inferência
- **`get_weights()`**: Recupera os pesos sinápticos e o viés ($a, b \text{ e } c$) aprendidos pela camada densa ao término do treinamento. ([Documentação Layer.get_weights](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_weights))
- **Chamada Direta do Modelo (`model(data, training=False).numpy()`)**: Executa a inferência diretamente pelo objeto do modelo em vez de usar `.predict()`, otimizando a velocidade e reduzindo overhead.

---

🚀 Próximos projetos  
Outros projetos serão postados em breve, com arquiteturas diferentes e níveis de complexidade maiores.

👤 Autor: Bruno Zucker
