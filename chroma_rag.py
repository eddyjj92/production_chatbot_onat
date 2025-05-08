import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from dotenv import load_dotenv

class DocumentStore:
    def __init__(self, documents: List[Document], embeddings_provider: str = "cloudflare"):
        """
        Inicializa la clase con los documentos proporcionados y configura el almacén vectorial Chroma.

        :param documents: Lista de documentos a almacenar.
        :param embeddings_provider: Proveedor de embeddings a utilizar.
        """
        # Cargar variables de entorno
        load_dotenv()
        self.documents = documents
        self.embeddings_provider = embeddings_provider
        self.collection_name = "onat_docs"
        self.persist_directory = "./chroma_db"
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self) -> Chroma:
        """
        Configura el almacén vectorial Chroma. Si el directorio persistente existe, carga la base de datos existente;
        de lo contrario, crea una nueva e ingesta los documentos proporcionados.

        :return: Instancia del almacén vectorial Chroma.
        """
        if self.embeddings_provider == "cloudflare":
            embeddings = CloudflareWorkersAIEmbeddings(
                model_name="@cf/baai/bge-large-en-v1.5",
                account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
                api_token=os.getenv("CLOUDFLARE_API_KEY"),
            )
        else:
            embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

        # Crear o cargar la base de datos
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            # Cargar la base de datos existente
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=embeddings
            )
        else:
            # Crear una nueva base de datos e ingestar documentos
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            doc_splits = text_splitter.split_documents(self.documents)
            vector_store = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=embeddings,
                persist_directory=self.persist_directory,
            )
            # No es necesario llamar a vector_store.persist()

        return vector_store

    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Recupera los documentos más relevantes basados en la consulta proporcionada.

        :param query: Consulta de búsqueda.
        :param k: Número de documentos relevantes a recuperar.
        :return: Lista de documentos relevantes.
        """
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.5, 'k': k},

        )
        return retriever.invoke(query)


initial_documents = [
    Document(page_content="La Oficina Nacional de Administración Tributaria (ONAT) es la entidad encargada de velar por la aplicación de la legislación relativa a impuestos y otros ingresos no tributarios en Cuba. Su misión incluye desarrollar la organización para su recaudación en todo el país y organizar y dirigir la auditoría fiscal, adoptando las medidas requeridas para contrarrestar la evasión fiscal. :contentReference[oaicite:0]{index=0}"),
    Document(page_content="Entre las funciones principales de la ONAT se encuentran: la gestión, control, determinación y recaudación de los tributos. :contentReference[oaicite:1]{index=1}"),
    Document(page_content="La ONAT brinda servicio de asistencia activa y personalizada a través de diferentes medios, durante y tras la apertura de los negocios, para garantizar el aporte correcto y en tiempo de las obligaciones tributarias. :contentReference[oaicite:2]{index=2}"),
    Document(page_content="La ONAT ha habilitado servicios online en su portal web, como el Vector Fiscal, que permite obtener el documento que los contribuyentes utilizan para pagar en el banco y que contiene todas sus obligaciones tributarias. :contentReference[oaicite:3]{index=3}"),
    Document(page_content="La ONAT se conecta con la Ficha Única del Ciudadano, lo que agiliza los trámites y servicios a los contribuyentes, minimiza errores en la captación de datos y evita solicitar la misma información varias veces. :contentReference[oaicite:4]{index=4}"),
    Document(page_content="La ONAT tiene como propósito cumplir el plan de recaudación, avanzar en su transformación digital e implementar el perfeccionamiento de la estructura administrativa y su funcionamiento. :contentReference[oaicite:5]{index=5}"),
    Document(page_content="La ONAT aplica intereses, recargos y sanciones que correspondan, tramita solicitudes de devoluciones de ingresos y emite certificaciones relacionadas con la situación fiscal de los contribuyentes. :contentReference[oaicite:6]{index=6}"),
    Document(page_content="La ONAT ha incrementado su presencia en redes sociales como Facebook, Twitter y el canal 'OnatdeCuba' en Telegram, ofreciendo servicios interactivos y facilitando la comunicación con los contribuyentes. :contentReference[oaicite:7]{index=7}"),
    Document(page_content="La ONAT es clave en el desarrollo económico, político y social del país, centrando su trabajo en ingresar al Presupuesto del Estado los ingresos definidos en la Ley del Presupuesto, de los cuales el 53% se capta como resultado de la aplicación de impuestos, tasas y contribuciones. :contentReference[oaicite:8]{index=8}"),
    Document(page_content="El Vector Fiscal es un documento emitido por la ONAT que detalla las obligaciones tributarias del contribuyente, incluyendo impuestos, tasas y contribuciones. Se actualiza automáticamente y puede descargarse desde el Portal Tributario. [Fuente: Desoft](https://www.desoft.cu/es/noticias/274)"),
    Document(page_content="Para obtener el Vector Fiscal, el contribuyente debe registrarse en el Portal Tributario de la ONAT. Este documento es esencial para realizar pagos en el banco y contiene todas las obligaciones fiscales del contribuyente. [Fuente: Facebook ONAT](https://www.facebook.com/onat.gob.cu/posts/si-necesita-descargar-su-vector-fiscal-debe-registrarse-en-el-portal-tributario-/682802034028919/)"),
    Document(page_content="La Declaración Jurada ( DJ ) es un documento oficial mediante el cual los contribuyentes informan a la Oficina Nacional de Administración Tributaria (ONAT) sobre los ingresos obtenidos durante un ejercicio fiscal, calculan el impuesto correspondiente y realizan el pago debido."),
    Document(page_content="La Declaración Jurada ( DJ ) del Impuesto sobre Ingresos Personales debe presentarse anualmente entre el 6 de enero y el 30 de abril. Los contribuyentes que declaren y paguen antes del 28 de febrero pueden acogerse a una bonificación del 5%. [Fuente: Granma](https://www.granma.cu/cuba/2024-12-28/comienza-el-6-de-enero-proceso-de-declaracion-jurada-de-onat)"),
    Document(page_content="La Ley 174 del Presupuesto del Estado para 2025 establece una nueva escala progresiva para el cálculo del Impuesto sobre Ingresos Personales, aplicable a los ingresos obtenidos en el ejercicio fiscal 2024. [Fuente: Juventud Rebelde](https://juventudrebelde.cu/cuba/2025-01-04/conozca-sobre-la-nueva-escala-progresiva-para-la-declaracion-jurada-del-impuesto-sobre-ingresos-personales)"),
    Document(page_content="Los modelos de Declaración Jurada, como el DJ 08, están disponibles en formato Excel y PDF en la sección Descargas del Portal Tributario de la ONAT. Estos modelos ayudan a los contribuyentes a calcular y presentar sus impuestos correctamente. [Fuente: Portal Tributario ONAT](https://www.onat.gob.cu/home/modelos-formularios)"),
    Document(page_content="La ONAT ha implementado servicios en línea que permiten a los contribuyentes consultar sus pagos realizados, descargar el Vector Fiscal y realizar consultas directamente desde el Portal Tributario. [Fuente: Instagram ONAT](https://www.instagram.com/onat_cuba/p/DCRaB1KqjOk/)"),
    Document(page_content="El proceso de declaración jurada para el ejercicio fiscal 2024 comienza el 6 de enero y concluye el 30 de abril de 2025. Los productores agropecuarios individuales del sector cañero deben presentar su declaración entre el 1 de julio y el 31 de octubre de 2025. [Fuente: Radio Progreso](https://www.radioprogreso.icrt.cu/onat-anuncia-proceso-de-declaracion-jurada-para-2025/)"),
    Document(page_content="La ONAT ofrece bonificaciones por pronto pago y por el uso de canales digitales. Declarar y pagar antes del 28 de febrero otorga un descuento del 5%, y utilizar canales digitales como Transfermóvil brinda un beneficio adicional del 3%. [Fuente: Artemisa Diario](https://artemisadiario.cu/2025/01/que-debe-saber-de-la-nueva-escala-progresiva-para-la-declaracion-jurada-del-impuesto-sobre-ingresos-personales/)"),
    Document(page_content="La url del Portal Tributario de la ONAT es : https://www.onat.gob.cu/"),

]