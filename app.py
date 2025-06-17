import ssl
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Get port from environment (Elastic Beanstalk uses PORT, but we'll also check for common alternatives)
port = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", 5000)))

# Validate required environment variables
required_env_vars = ["TAVILY_API_KEY", "GROQ_API_KEY", "MONGO_URI"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

# Set environment variables
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Initialize MongoDB connection with error handling
try:
    client = MongoClient(MONGO_URI)
    # Test connection
    client.admin.command('ping')
    db = client["project2"]
    collection = db["vectors2"]
    logger.info("Successfully connected to MongoDB Atlas")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# Initialize embedding model
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    raise

# Initialize MongoDB Atlas vector store
try:
    book_db = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_model,
        index_name="vector_index2",
        connection_kwargs={"tls": True}
    )
    logger.info("MongoDB Atlas vector store initialized")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    raise

# Define MongoDB retrieval tool
def search_mongo_tool(query: str):
    try:
        docs = book_db.similarity_search(query, k=2)
        return "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant book content found."
    except Exception as e:
        logger.error(f"MongoDB search error: {e}")
        return "Error searching medical database."

# Define Tavily web search tool
def search_tavily_tool(query: str):
    try:
        retriever = TavilySearchAPIRetriever(k=5)
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant web content found."
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return "Error searching web content."

# Define tools for the agent
tools = [
    Tool(
        name="MedicalVectorRetriever",
        func=search_mongo_tool,
        description="Search the medical book using patient symptoms and vitals."
    ),
    Tool(
        name="WebSearchRetriever",
        func=search_tavily_tool,
        description="Search the web when the book lacks relevant info."
    )
]

# Initialize Groq LLM
try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile"
    )
    logger.info("Groq LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {e}")
    raise

# Initialize LangChain agent
try:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    logger.info("LangChain agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    raise

def parse_agent_output(text_data):
    """Parse agent output into structured JSON format"""
    result = {"diseases": [], "tests": []}

    if "POSSIBLE DISEASES:" in text_data and "DIAGNOSTIC TESTS:" in text_data:
        diseases_part = text_data.split("DIAGNOSTIC TESTS:")[0].replace("POSSIBLE DISEASES:", "").strip()
        tests_and_tips_part = text_data.split("DIAGNOSTIC TESTS:")[1]

        if "TIPS:" in tests_and_tips_part:
            tests_part = tests_and_tips_part.split("TIPS:")[0].strip()
        else:
            tests_part = tests_and_tips_part.strip()

        # Parse diseases
        for line in diseases_part.split('\n'):
            if line.strip().startswith("-"):
                parts = line.strip().lstrip("- ").split(" - ", 1)
                if len(parts) == 2:
                    name, description = parts
                    result["diseases"].append({
                        "name": name.strip(),
                        "description": description.strip()
                    })

        # Parse tests
        for line in tests_part.split('\n'):
            if line.strip().startswith("-"):
                parts = line.strip().lstrip("- ").split(" - ")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    description = parts[1].strip() if len(parts) > 1 else "No description"
                    tips = parts[2].strip() if len(parts) > 2 else ""
                    
                    result["tests"].append({
                        "name": name,
                        "description": description,
                        "tips": tips
                    })
    else:
        result["tests"].append({
            "name": "No diagnostic output",
            "description": "Agent output did not match expected format.",
            "tips": ""
        })

    return result

# Health check endpoint for AWS load balancer
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "medical-diagnostic-api"}), 200

# Main API endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid JSON payload"
            }), 400

        symptoms = data.get("symptoms", "").strip()
        vitals = data.get("vitals", "").strip()
        age = data.get("age", "").strip()
        
        # Validate required fields
        if not symptoms or not vitals or not age:
            return jsonify({
                "success": False,
                "error": "Missing required fields: age, symptoms, or vitals."
            }), 400

        # Create the prompt
        prompt = f"""
You are a medical assistant AI that provides:

1. Possible diseases with one-line descriptions based on the patient's age, symptoms, and vitals.
2. Diagnostic test recommendations with one-line justifications considering the patient's specific profile, and additional suitability tips.

Rules:
- If the input does NOT contain valid medical symptoms, vitals, or age, respond with:
  "I cannot provide diagnostic test recommendations or disease names without valid medical symptoms, vitals, and age."
- For valid inputs, follow this output format exactly:

POSSIBLE DISEASES:
- Disease Name - One line description.
- Disease Name - One line description.

DIAGNOSTIC TESTS:
- Test Name - Short reason - Suitability tips or contraindications.

Input query:
Patient Age: {age}. Symptoms: {symptoms}. Vitals: {vitals}
"""

        logger.info(f"Processing request for patient age: {age}")
        
        # Run the agent
        agent_output = agent.run(prompt)
        parsed_output = parse_agent_output(agent_output)

        logger.info("Successfully processed diagnostic request")
        
        return jsonify({
            "success": True,
            "diseases": parsed_output["diseases"],
            "tests": parsed_output["tests"]
        })

    except Exception as e:
        logger.error(f"Agent Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "An internal error occurred. Please try again later."
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info(f"Starting Flask app on port {port}")
    # For production, use a proper WSGI server like Gunicorn
    app.run(host="0.0.0.0", port=port, debug=False)