# Etapa 8 Mostrar os resultados: Imprimir os pesos, vieses e o gasto previsto.

# Função para mostrar os resultados na tela, incluindo os pesos e vieses aprendidos pelo modelo, bem como o gasto previsto para a viagem.
def mostrar_resultados(carga, distancia, perdas, pesos, vies, gasto_previsto):
    
    # Mostrar os valores inseridos pelo usuário
    print("\nValores inseridos: Carga:", carga, "kg, Distância:", distancia, "km")

    # O MSE (Mean Squared Error) mede a média dos erros ao quadrado. 
    # Calculando a raiz quadrada do loss final (RMSE = sqrt(0.68) ≈ 0.82), 
    # o modelo apresenta um erro médio inferior a 1% na previsão do gasto de bateria.
    print("\nLoss final (MSE):", perdas)

    # Mostra a carga com os pesos
    print("\na (carga kg):", pesos[0][0])

    # Mostra a distância com os pesos
    print("b (distância km):", pesos[1][0])

    # Mostra o viés do modelo
    print("c (viés):", vies[0])

    # Mostra o gasto previsto pela IA para os valores inseridos pelo usuário
    print("\nGasto previsto %:", gasto_previsto[0][0])