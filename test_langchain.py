import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Union
from langchain_core.messages import BaseMessage, HumanMessage

# ===== World simulation =====
class Mundo:
    def __init__(self) -> None:
        self.bateria = 100
        self.posicao: tuple[int, int] = (0, 0)
        self.obstaculos = [(1, 1), (1, 3), (3, 1)]
        self.destino = (5, 5)

mundo = Mundo()

# ===== Tools definition =====
@tool
def consultar_sensores():
    """Consulta o nível da bateria, posição atual e obstáculos próximos."""
    return {
        "bateria": mundo.bateria,
        "posicao": mundo.posicao,
        "obstaculos": mundo.obstaculos,
        "destino": mundo.destino
    }

@tool
def mover_para(x: int, y: int):
    """
    Move o drone para a posição (x, y).
    Cada movimento consome 5% da bateria.
    """

    if (x, y) in mundo.obstaculos:
        return "Movimento bloqueado por obstáculo."
    if mundo.bateria < 5:
        return "Bateria insuficiente para mover."
    
    mundo.posicao = (x, y)
    mundo.bateria -= 5
    return f"Movido para ({x}, {y}). Bateria restante: {mundo.bateria}%."

tools = [consultar_sensores, mover_para]

# ===== Agent state definition =====
class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str

# ===== Model configuration =====
llm = ChatOllama(
    model="qwen3.5:4b",
    base_url="http://localhost:11434", 
    temperature=0
)

# ===== Agents and State definition =====
def drone_node(state: AgentState):
    """Agente Drone: Toma decisões de navegação."""
    llm_with_tools = llm.bind_tools([consultar_sensores, mover_para])
    prompt = "Você é o Agente Drone. Use as ferramentas disponíveis para navegar até o destino, evitando obstáculos e gerenciando a bateria."
    response = llm_with_tools.invoke([HumanMessage(content=prompt)] + list(state["messages"]))
    return {"messages": [response]}

# ===== Agent definition =====
drone_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "Você é um drone autonomo. Sua missão é entregar um pacote no destino informado. "
        "Evite obstáculos e gerencie sua bateria (cada movimento consome 5%)."
    ),
)

wind_agent = create_agent(
    model=llm,
    tools=[mover_para],
    system_prompt=(
        "Você é um agente especializado em simular condições climáticas adversas."
    ),
)

# ===== Execution =====
print("Iniciando o agente de entrega...")

resultado = drone_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Entregue o pacote no destino {mundo.destino}. "
                    "Liste os passos realizados."
                ),
            }
        ]
    }
)

print("Resultado final da missão:")
print(resultado["messages"][-1].content)
