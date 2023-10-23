# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://datasolut.com/wp-content/uploads/2020/01/logo-horizontal.png" alt="Databricks Learning" style="width: 300px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # <img src="https://yt3.googleusercontent.com/ytc/AL5GRJWSDkfSdYxhsFknPhzJUWjYkHMEdYJHRO_AuCzlfQ=s900-c-k-c0x00ffffff-no-rj" alt="ds Logo Tiny" width="100" height="100"/> LLM Demo: Retrieval Augmented Generation 
# MAGIC
# MAGIC In diesem Notebook demonstrieren wir die Verwendung von LLMs zur Retrieval Augmented Generation (RAG).
# MAGIC Dabei wird ein Information Retrieval Verfahren mit einem Generativen Sprachmodell kombiniert, um Text zu Themen zu generieren, die tiefes Domänenwissen erfordern - auf das die üblichen LLMs als Generalisten nicht spezialisiert sind.
# MAGIC
# MAGIC Besuchen Sie auch gerne die [datasolut-Website](https://datasolut.com/) und unseren [Youtube-Kanal](https://www.youtube.com/@datasolut6213)!
# MAGIC
# MAGIC <img src="files/yt_files/RAG_erklaerung.jpg" alt="drawing" style="width:800px;"/>
# MAGIC
# MAGIC Das Domänenwissen sind in diesem Beipiel alle Texte, die auf unserem Blog und unserer Webseite zu finden sind. Wir nutzen die Library BeautifulSoup um die Inhalte der Webseite zu scrapen.
# MAGIC
# MAGIC ... Korpus, etc...
# MAGIC
# MAGIC <img src="files/yt_files/RAG_Uebersicht.jpg" alt="drawing" style="width:800px;"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup: 
# MAGIC - Dependencies installieren
# MAGIC - Credentials einlesen

# COMMAND ----------

# dependencies for llms
!pip install tiktoken langchain faiss-cpu openai sqlalchemy==1.4.49 -q
# dependencies for webscraper
!pip install requests beautifulsoup4 -q

# COMMAND ----------

import os
openai_api_key = os.getenv("OPENAI_API_KEY")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Webscraping des datasolut Blogs
# MAGIC Um unseren Blog zu scrapen können wir auf die Blog-Sitemap unserer Domain zugreifen und alle Links extrahieren. Den Inhalt können wir mit dem BeautifulSoup Package scrapen.

# COMMAND ----------

from bs4 import BeautifulSoup
import os
import re
import requests

myURL = "https://datasolut.com/post-sitemap.xml"
page = requests.get(myURL)
soup = BeautifulSoup(page.content, 'html.parser')

# alle Links extrahieren und in einer Liste speichern
blog_links = []
for link in soup.find_all('loc'):
    path = link.get_text()
    blog_links.append(path)

# COMMAND ----------

# check blog_links
blog_links[2]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dokumente laden
# MAGIC Um die Blog-Inhalte in Langchain-kompatible Documents zu überführen, kann man einen der vielen Document-Loader von Langchain verwenden.

# COMMAND ----------

from langchain.document_loaders import WebBaseLoader

documents = []
loader = WebBaseLoader(blog_links)
scrape_data = loader.load()

# COMMAND ----------

print(f"total documents scraped: {len(scrape_data)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Postprocessing der Dokumente
# MAGIC Ohne Postprocessing enthalten die Blogs viele irrelevante Links und Informationen. Beispielsweise sind die Navigationsleiste und der Footer immer enthalten. Diese möchten wir nicht mit den im Blog besprochenen Themen vermischen. Außerdem möchten wir die Token-Länge der Blog-Einträge auf ein Minimum reduzieren, ohne dabei den Inhalt zu verlieren. Deshalb wurde für diesen Teil ein Start- und ein Endmarker im html identifiziert, wo der Blog-Inhalt für jeden Link anfängt und aufhört. Desweiteren werden bestimmte unicode-characters wie '\xa0' (space) ersetzt.

# COMMAND ----------

start_identifier = 'KI-Projekt starten'
end_identifier = 'Ihr Kontakt: Laurenz Wuttke'

for d in scrape_data:
    d.page_content = d.page_content.split(start_identifier)[1].split(end_identifier)[0].strip().replace(u'\xa0', u' ')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunking
# MAGIC Das Chunking ist ein wichtiger Teil der Aufbereitung der Texte. Dies liegt an der unvermeidbaren Token-Limitation von LLM's. Diese liegt meist bei 8000 (z.B. gpt-3.5) oder 16000 (z.B. gpt-4).
# MAGIC
# MAGIC Hintergrund ist, dass ein LLM Text als Kontext überliefert bekommt neben der gestellten Frage (was ist Churn Prevention?). Dieser Kontext kommt aus unserer internen Wissens-Datenbank.
# MAGIC
# MAGIC Sind die einzelnen Dokumente also 10000 Tokens lang (Token ist hier nicht gleichzusetzen mit Wort sondern kleinteiliger), bleibt dem in der LLM-Anfrage kein Platz für die gestellte Frage. Ist ein Dokument allerdings nur 400 Tokens lang, kann dem LLM der Kontext, allgemeine Systemnachrichten (you are a helpful chatbot ...), die Frage sowie auch die bisherige Konversation übermittelt werden.
# MAGIC
# MAGIC <img src="files/yt_files/RAG_Uebersicht.jpg" alt="drawing" style="width:800px;"/>
# MAGIC
# MAGIC Hier gibt es immer einen Trade-Off:
# MAGIC     
# MAGIC längere Chunks erlauben dem LLM ausführlichere Informationen zu generieren und komplexere Zusammenhänge zu verstehen, es besteht jedoch die Gefahr, dass für die gestellte Frage auch unwichtige Informationen generiert werden und die Frage unnötig umfassend/komplex beantwortet
# MAGIC     
# MAGIC bei kürzeren Chunks kann nur die relevante Information aus der eigenen Datenbank gefunden und übermittelt werden und die kosten sind aufgrund geringerer Token kleiner aber komplexere Zusammenhänge oder längere Erklärungen aus der eigenen Datenbank sind eventuell in unterschiedliche Dokumente unterteilt. Um dem entgegenzuwirken definiert man einen Chunk-overlap, sodass das Ende des vorigen Chunks immer auch am Anfang des folgenden Chunks vorhanden ist.

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
print(f"Using encoding: {encoding}")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap  = 100,
    length_function = num_tokens_from_string,
    is_separator_regex = False,
)

documents = []
for d in scrape_data:
    chunks = text_splitter.split_text(d.page_content)
    for i, chunk in enumerate(chunks):
        documents.append(Document(page_content=chunk, metadata={'chunk': i, **d.metadata}))

print(f"Total number of documents: {len(documents)}")
print(f"""\nsample page content:
      \n{documents[1].page_content}
      """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Erzeuge Vectorstore
# MAGIC In diesem Abschnitt wird eine Datenbank für enorm schnelle Abfragen gebaut und persistiert. Für einen kleinen Korpus wie in diesem Fall wäre ein einfacher Pandas Dataframe jedoch auch völlig ausreichend.

# COMMAND ----------

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import pickle

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

with open("vectorstore_blog.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Erzeuge LangChain Agent
# MAGIC In diesem Abschnitt wird der eigentliche LLM-Agent definiert und aufgesetzt. Ein retriever ist sozusagen verantwortlich für das Auffinden der n wichtigsten Dokumente/Chunks für eine jeweilige Abfrage/Query, damit diese in der Folge dem LLM als Kontext übergeben werden kann.
# MAGIC
# MAGIC Eine Abfrage besteht also immer aus mehreren API-Calls. Dies ist wichtig zu beachten, da mit jeder Abfrage Kosten generiert werden.
# MAGIC Die erste Abfrage kann man sich vorstellen wie: "dir stehen folgende Optionen zur Auswahl, um die folgende Frage zu beantworten: 
# MAGIC
# MAGIC 1. generiere sofort eine Antwort; 
# MAGIC 2. suche in search_datasolut_blog nach relevanten Informationen; Frage: was ist Churn Prevention?"
# MAGIC
# MAGIC Dies ist absolut nicht der Wortlaut, wie es bei beispielsweise ChatGPT funktioniert, aber so in der Art kann man es sich vorstellen.
# MAGIC Der Output dieser Abfrage ist also eine Entscheidung, ob direkt (auf bereits bestehendem Wissen oder basierend auf der bisherigen Chat-Historie) geantwortet werden kann, oder ob die eigene Datenbank abgefragt werden soll.
# MAGIC Die Definition/Beschreibung des Tools ist dabei Entscheidend, wann und wieso der Agent auf das Tool zugreifen soll.
# MAGIC
# MAGIC Wird die eigene Datenbank abgefragt, wird eine einfache Ähnlichkeitssuche der Embeddings durchgeführt (auch die Frage muss also von dem Embedding-Modell, welches von gpt-3.5/4 genutzt wird in Vektoren umgewandelt werden -> so sind auch die eigenen Dokumente in der Datenbank bereits durch das Embedding-Modell gelaufen).
# MAGIC Die Top N Dokumente werden dann in einem nächsten API-Call dem LLM als Kontext zusammen mit der Frage übergeben.
# MAGIC Dies sieht dann ungefähr so aus:
# MAGIC "Basierend auf gegebenem Kontext <Kontext1 Kontext2 Kontext3> beanworte die folgende Frage <Frage>."

# COMMAND ----------

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI

retriever = vectorstore.as_retriever()

tool = create_retriever_tool(
    retriever, 
    "search_datasolut_blog",
    "Searches and returns documents regarding the Datasolut GmbH blog." # Die Definition/Beschreibung des Tools ist dabei Entscheidend, wann und wieso der Agent auf das Tool zugreifen soll.
)
tools = [tool]

llm = ChatOpenAI(temperature = 0)
agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

# COMMAND ----------

# hier ist die Entscheidung gefallen, dass die eigene Datenbank nicht konsultiert werden muss
result = agent_executor({"input": "hi, im bob"})

# COMMAND ----------

# hier erkennt der Agent jedoch, dass die Datenbank abgefragt werden sollte; im folgenden Output (den man übrigens unterdrücken kann) sieht man, dass die 4 relevantesten Dokumente übergeben worden sind
result = agent_executor({"input": "Welche Herausforderungen bestehen im Rahmen der Kundenklassifizierung?"})

# COMMAND ----------

# nur auf den output zugreifen
print(result['output'])

# COMMAND ----------

# da sich in dieser Frage nur auf die bisherige Historie bezogen wird, erkennt der Agent, dass diese als Kontext ausreichend ist -> es wird standardmäßig auch immer der bisherige Chatverlauf übermittelt
# interessant ist hier auch, dass Metadaten, wie der Link eines Dokuments erfolgreich übermittelt und erkannt werden
result = agent_executor({"input": "Wie lautet der link dieses blog posts?"})

# COMMAND ----------

# auch bei einer weiteren Folgefrage können die richtigen Metadaten extrahiert werden
result = agent_executor({"input": "welcher chunk ist das?"})

# COMMAND ----------

result = agent_executor({"input": "was ist der genaue Text dieses chunks?"})

# COMMAND ----------

result = agent_executor({"input": "Was ist Churn Prevention?"})

# COMMAND ----------

print(result['output'])

# COMMAND ----------

result = agent_executor({"input": "Was sind die Vorteile von einem Data Lake?"})

# COMMAND ----------

result = agent_executor({"input": "Fasse mir die 3 wichtigsten Vorteile eines Data Lakes laut unseres Blog zusammen."})

# COMMAND ----------

result = agent_executor({"input": "Was ist der Unterschied zwischen einem Data Lake und einem Data Lakehouse?"})

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Text: Diese Lösung kann nun z.B. per REST API gehostet werden.
# MAGIC Oder als MlFlow Python Model? Noch klären...
