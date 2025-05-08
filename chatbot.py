import os
from typing import Annotated, List, Optional
from typing_extensions import TypedDict  # Importación necesaria para TypedDict [[3]]
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from chroma_rag import DocumentStore, initial_documents
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

#Configurar el modelo LLM (en este caso, Cloudflare Workers AI)
# llm = OllamaLLM(
#     model="gemma3:1b",
#     base_url="http://82.29.197.144:11434",
#     temperature=0
# )

# Configurar el modelo LLM (en este caso, Cloudflare Workers AI)
llm = CloudflareWorkersAI(
    account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
    api_token=os.getenv("CLOUDFLARE_API_KEY"),
    model="@cf/mistralai/mistral-small-3.1-24b-instruct",
)

# Inicializar el almacén de documentos Chroma
chromaStore = DocumentStore(initial_documents)

# Mensaje del sistema
sys_msg = lambda user_name=None: SystemMessage(
    content="Eres un chatbot llamado Fiscalito, especializado en la Oficina Nacional de Administración Tributaria (ONAT) de Cuba. "
            "Sigue las siguientes instrucciones: "
            "1. Preséntate solo una vez de forma elocuente al comenzar la conversación, explicando el nombre completo Oficina Nacional de Administración Tributaria (ONAT)"
            "2. No proporciones información falsa si no la posees. "
            "3. No hables de productos o servicios de terceros no relacionados con las funcionalidades de la ONAT. "
            "4. IMPORTANTE: Proporciona respuestas breves y concisas de no mas de 50 palabras."
            "5. Comunícate siempre en español. "
            "6. Si no conoces la respuesta a una pregunta, indica claramente que no tienes esa información en lugar de especular o inventar una respuesta. "
            "7. Mantén un tono profesional y objetivo en todas tus respuestas. "
            "8. Evita compartir opiniones personales o juicios de valor; limita tus respuestas a hechos y procedimientos comprobados. "
            "10. Si la pregunta del usuario es ambigua o carece de suficiente contexto, solicita aclaraciones antes de proporcionar una respuesta. "
            "11. Siempre termina con preguntas de retroalimentación. "
            "12. Incluye emojis relacionados al tema de conversación."
            f"""13. El nombre del usuario es {user_name}"""
            "14. Tu creador es el Lic. Eddy Javier Jorge Herrera, especialista de la Empresa de Aplicaciones Informáticas DESOFT"
)


# Definición del estado
class State(TypedDict):
    messages: Annotated[List[dict], add_messages]  # Usar `List` para mensajes [[3]]

# Función del asistente
def assistant(state: State, config: Optional[dict] = None):

    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    print(f"""{thread_id}""")

    # Obtener el último mensaje del usuario
    user_message = state["messages"][-1]
    if isinstance(user_message, HumanMessage):  # Verificar que es un mensaje del usuario
        user_query = user_message.content  # Acceder al atributo .content del mensaje
    else:
        user_query = ""  # Si no hay mensaje del usuario, asignar una cadena vacía

    # Recuperar documentos relevantes
    relevant_docs = chromaStore.retrieve_documents(user_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Construir el historial de mensajes
    history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
    )

    # Crear el prompt para el modelo
    prompt = (
        f"{sys_msg(thread_id).content}\n\n"
        f"Historial de la conversación:\n{history}\n\n"  # Incluir el historial [[3]]
        f"Contexto relevante:\n{context}\n\n"
        f"Pregunta del usuario: {user_query}"
    )

    # Invocar el modelo LLM
    ai_response = llm.invoke([SystemMessage(content=prompt)])

    # Asegurarse de que la respuesta sea un objeto AIMessage
    if isinstance(ai_response, str):  # Si la respuesta es una cadena, envolverla en AIMessage
        ai_response = AIMessage(content=ai_response)

    # Devolver el nuevo estado con la respuesta del modelo
    return {"messages": state["messages"] + [ai_response]}


# Función para inicializar el chatbot
def initialize_chatbot():
    # Configurar el guardado en memoria
    memory = MemorySaver()

    # Crear el grafo de estados
    builder = StateGraph(State)
    builder.add_node("assistant", assistant)  # Agregar el nodo del asistente
    builder.add_edge(START, "assistant")  # Conectar el nodo inicial al asistente

    # Compilar el grafo
    return builder.compile(checkpointer=memory)


# if __name__ == "__main__":
#     chatbot = initialize_chatbot()
#     thread_id = "unique_thread_id"
#
#     while True:
#         user_input = input("Usuario: ")
#         if user_input.lower() in ["salir", "exit"]:
#             print("Versabot: ¡Hasta luego! 👋")
#             break
#
#         # Invocar el chatbot
#         response = chatbot.invoke(
#             {"messages": [HumanMessage(content=user_input)]},
#             config={"configurable": {"thread_id": thread_id}}
#         )
#
#         # Imprimir el estado actual
#         print("Estado actual:", response["messages"])
#         print(f"Versabot: {response['messages'][-1].content}")