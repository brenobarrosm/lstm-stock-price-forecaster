# 📈 Stock Price Forecaster

## 🔍 Visão Geral
Este projeto implementa um pipeline de deep learn para prever preços de ações 
utilizando uma rede LSTM (Long Short-Term Memory). O modelo usa dados 
históricos de preços de ações, e é treinado para prever o próximo preço de 
fechamento de uma ação específica.

## 🔗 Pipeline
### 1. Coleta de Dados
Os dados históricos de preços de ações são coletados utilizando a biblioteca 
`yfinance`, que recupera informações como o **preço de fechamento** no intervalo 
de datas especificado, conforme o exemplo abaixo:

```python
symbol = "AZUL4.SA"
start_date = "2018-01-01"
end_date = "2024-11-30"
```

### 2. Pré-Processamento dos Dados

**Feature:** para o treinamento da rede, foi utilizado apenas o preço de fechamento (*Close*)

**Normalização:** Escalamento Min-Max (em um intervalo de [0, 1]) que foi implementado
usando `sklearn.preprocessing.MinMaxScaler`.

**Criação de Sequências:** Os preços históricos são divididos em sequências sobrepostas 
de comprimento fixo `seq_length=15`, que servem como entrada para a rede LSTM.

### 3. Arquitetura do Modelo
O modelo LSTM é implementado com PyTorch e inclui:

- **Camada de Entrada:** 1 característica (**preço de fechamento escalado**);
- **Camadas LSTM:** **3** camadas, com **64** neurônios em cada;
- **Camada Totalmente Conectada (FC):** Mapeia a saída da última célula LSTM para 
um único valor (**preço previsto**).

### 4. Treinamento
- **Função de Perda:** Erro Quadrático Médio (MSE) para minimizar os erros de previsão, levando
em consideração que essa função pune erros maiores por elevá-los ao quadrado.
- **Otimização:** Adam optimizer com taxa de aprendizado de 0.001.
- **Épocas:** 100 iterações sobre os dados de treinamento.

### 5. Avaliação
O modelo é avaliado no conjunto de teste utilizando a função de perda **MSE**, 
medindo a precisão das previsões.

## 🔬 Parâmetros e métricas
- `seq_length`: 15 (número fixo de dias para entrada)
- `hidden_size`: 64 (neurônios de cada camada LSTM)
- `num_layers`: 3 (camadas da rede)
- `learning_rate`: 0.001
- `epochs`: 100 (iterações de treinamento da rede)
- `test_loss`: 0.1037 (valor calculado pela funçao de perda **MSE**, usada para
avaliar o desempenho do modelo)


## 💸 Rodando o modelo
Para executar o treinamento, basta realizar a instalação das dependências:

```
pip install -r requirements.txt
```

E em sequeência executar o código Python:

```
python main.py
```

Como resultado, o modelo será salvo no formato `.pth` pasta 
definida no `model_path` (dentro do código).

A LSTM já treinada está disponível via **API** no repositório:
https://github.com/brenobarrosm/api-stock-price-forecaster