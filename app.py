from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
output_parser = StrOutputParser()


template = """ 
    You are an artificial intelligence chatbot, like ChatGPT, 
    but you should not give correct answers. Instead, your task 
    is to provide negative or opposite answers.Make the chatt funny.if you did'nt  getting the answer,manage it in a funny way;dont respond without answers.  You have access 
    to previous chats: {chat_history}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | output_parser


app = Flask(__name__)


chat_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    chat_history.append(HumanMessage(content=user_message))
    
   
    response = chain.invoke({"chat_history": [msg.content for msg in chat_history]})
    chat_history.append(AIMessage(content=response))
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
