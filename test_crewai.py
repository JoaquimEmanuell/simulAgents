import os

# Desativa a telemetria do CrewAI antes da biblioteca ser importada.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_STORAGE_DIR"] = os.path.join(os.getcwd(), ".crewai")
os.makedirs(os.environ["CREWAI_STORAGE_DIR"], exist_ok=True)

from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool

os.environ["OLLAMA_HOST"] = "http://localhost:11434"

llm_ollama = LLM(
    provider="ollama",
    model="qwen3.5:4b",
    base_url="http://localhost:11434",
    api_key="ollama",
)

# Simulação do mundo
class Mundo:
    def __init__(self) -> None:
        self.bateria = 100
        self.posicao:tuple[int, int] = (0, 0)
        self.obstaculos = [(1, 1), (1, 3), (3, 1)]
        self.destino = (5, 5)

mundo = Mundo()

@tool("consultar_sensores")
def consultar_sensores():
    """Consulta o nível da bateria, posição atual e obstáculos próximos."""
    return {
        "bateria": mundo.bateria,
        "posicao": mundo.posicao,
        "obstaculos": mundo.obstaculos,
        "destino": mundo.destino
    }

@tool("mover_para")
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

drone = Agent(
    role="Drone",
    goal="Andar pelo espaço sem esabarrar em obstaculos e sem deixar a bateria acabar no meio do voo",
    backstory="""Você é um drone autonomo que tem a missão de explorar o espaço aéreo. 
    Você deve navegar por diferentes ambientes, evitando obstáculos e gerenciando sua bateria para garantir um voo seguro e eficiente.""",
    tools=[consultar_sensores, mover_para],
    llm=llm_ollama,
    verbose=True,
    allow_delegation=False
)

tarefa_entrega = Task(
    name="Entrega de Pacote",
    description="Entregar um pacote para um local específico dentro do espaço aéreo designado.",
    expected_output="Confirmação de entrega bem-sucedida, incluindo localização e timestamp.",
    agent=drone
)

crew = Crew(
    name="Equipe de Entrega Aérea",
    agents=[drone],
    tasks=[tarefa_entrega],
    verbose=True
)

# Execução
print("== Iniciando a missão de entrega ==")
resultado = crew.kickoff()

print("== Resultado da missão ==")
print(resultado)
