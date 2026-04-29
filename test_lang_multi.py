import os
import random
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage

# ===== World simulation =====
class Mundo:
    def __init__(self) -> None:
        self.bateria = 100
        self.posicao: tuple[int, int] = (0, 0)
        self.obstaculos = [(1, 1), (1, 3), (3, 1)]
        self.destino = (5, 5)
        self.vento_ativo = False

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
    Consome bateria. Se houver vento, pode ser deslocado aleatoriamente e consome mais bateria.
    """

    if (x, y) in mundo.obstaculos:
        return "Movimento bloqueado por obstáculo."
    
    custo = 5

    if mundo.vento_ativo:
        custo = 8 
        dx, dy = random.choice([
            (0,1), (0,-1), (1,0), (-1,0)
        ])
        x += dx
        y += dy

    if mundo.bateria < custo:
        return "Bateria insuficiente para mover."
    
    mundo.posicao = (x, y)
    mundo.bateria -= custo

    return (
        f"Movido para ({x}, {y}). "
        f"Bateria restante: {mundo.bateria}%. "
        f"{'Vento afetou o movimento.' if mundo.vento_ativo else ''}"
    )

@tool
def ventar(ativo: bool):
    """Ativa ou desativa o vento, que pode afetar a navegação do drone."""
    mundo.vento_ativo = ativo
    estado = "ativo" if ativo else "parado"
    return f"Vento agora está {estado}."

tools = [consultar_sensores, mover_para, ventar]

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
    tools=[ventar],
    system_prompt=(
        "Você é um agente climático. "
        "Decida quando o vento deve começar ou parar de forma natural. "
        "Evite mudanças bruscas o tempo todo. "
        "Às vezes mantenha o vento, às vezes altere."
    ),
)

class AgentState(TypedDict):
    messages: list[BaseMessage]

def drone_step(state: AgentState):
    response = drone_agent.invoke(state)
    return {"messages": state["messages"] + [response["messages"][-1]]}

def wind_step(state: AgentState):
    response = wind_agent.invoke(state)
    return {"messages": state["messages"] + [response["messages"][-1]]}

graph = StateGraph(AgentState)

graph.add_node("drone", drone_step)
graph.add_node("vento", wind_step)

# fluxo alternado simples
graph.set_entry_point("vento")

graph.add_edge("vento", "drone")
graph.add_edge("drone", "vento")

# condição de parada simples
def should_end(state: AgentState):
    if mundo.posicao == mundo.destino:
        return END
    if mundo.bateria <= 0:
        return END
    return "vento"

graph.add_conditional_edges("drone", should_end)

app = graph.compile()

# ===== Execution =====
print("Iniciando o agente de entrega...")

resultado = app.invoke({
    "messages": [
        HumanMessage(
            content=f"Entregue o pacote no destino {mundo.destino}."
        )
    ]
})

print("Resultado final:")
print(resultado["messages"][-1].content)
