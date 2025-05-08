import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from chatbot import initialize_chatbot, chromaStore
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

# Inicialización de la aplicación FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lista de orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados
)

react_graph_memory = initialize_chatbot()


# Modelo Pydantic para la solicitud
class QueryRequest(BaseModel):
    query: str
    user_name: str
    k: int = 3

# Ruta para recuperar documentos relevantes
@app.get("/retrieve_documents/{query}/{k}")
async def retrieve_documents(query: str, k: int):
    try:
        results = chromaStore.retrieve_documents(query, k)
        return {"documents": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        messages = react_graph_memory.invoke(
            {"messages": [HumanMessage(content=request.query)]},
            config={"configurable": {"thread_id": request.user_name}}
        )
        messages["user_name"] = request.user_name

        texto = messages["messages"][-1].content
        parte_deseada = texto.split("Fiscalito: ")[-1]
        return {
            "reply": parte_deseada,
            "history": messages["messages"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
