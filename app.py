from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import ssl
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
required_env_vars = ["TAVILY_API_KEY", "GROQ_API_KEY", "MONGODB_URI"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

# Set environment variables
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

# Initialize MongoDB connection with SSL fix
try:
    # Multiple connection methods to handle SSL issues
    connection_methods = [
        # Method 1: TLS with invalid certificates allowed (most likely to work)
        {
            "tls": True,
            "tlsAllowInvalidCertificates": True,
            "tlsAllowInvalidHostnames": True,
            "serverSelectionTimeoutMS": 15000,
            "connectTimeoutMS": 15000,
            "socketTimeoutMS": 15000
        },
        # Method 2: SSL with CERT_NONE
        {
            "ssl": True,
            "ssl_cert_reqs": ssl.CERT_NONE,
            "serverSelectionTimeoutMS": 15000,
            "connectTimeoutMS": 15000,
            "socketTimeoutMS": 15000
        },
        # Method 3: Custom SSL context
        {
            "ssl": True,
            "ssl_context": ssl.create_default_context(),
            "serverSelectionTimeoutMS": 15000
        }
    ]
    
    client = None
    connection_error = None
    
    for i, kwargs in enumerate(connection_methods, 1):
        try:
            logger.info(f"Attempting MongoDB connection method {i}")
            if "ssl_context" in kwargs:
                kwargs["ssl_context"].check_hostname = False
                kwargs["ssl_context"].verify_mode = ssl.CERT_NONE
            
            client = MongoClient(MONGODB_URI, **kwargs)
            # Test connection
            client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB using method {i}")
            break
        except Exception as e:
            connection_error = e
            logger.warning(f"MongoDB connection method {i} failed: {e}")
            if client:
                client.close()
                client = None
    
    if not client:
        logger.error(f"All MongoDB connection methods failed. Last error: {connection_error}")
        raise connection_error
    
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
        connection_kwargs={"tls": True, "tlsAllowInvalidCertificates": True}
    )
    logger.info("MongoDB Atlas vector store initialized")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    raise

# Define MongoDB retrieval tool
def search_mongo_tool(query: str):
    """Search medical database for relevant information"""
    try:
        docs = book_db.similarity_search(query, k=2)
        if docs:
            content = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Found {len(docs)} relevant documents in medical database")
            return content
        else:
            logger.info("No relevant medical documents found")
            return "No relevant book content found."
    except Exception as e:
        logger.error(f"MongoDB search error: {e}")
        return "Error searching medical database."

# Define Tavily web search tool
def search_tavily_tool(query: str):
    """Search web for additional medical information"""
    try:
        retriever = TavilySearchAPIRetriever(k=5)
        docs = retriever.get_relevant_documents(query)
        if docs:
            content = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Found {len(docs)} relevant web documents")
            return content
        else:
            logger.info("No relevant web content found")
            return "No relevant web content found."
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return "Error searching web content."

# Define tools for the agent
tools = [
    Tool(
        name="MedicalVectorRetriever",
        func=search_mongo_tool,
        description="Search the medical book database using patient symptoms and vitals to find relevant medical information."
    ),
    Tool(
        name="WebSearchRetriever",
        func=search_tavily_tool,
        description="Search the web for additional medical information when the medical database lacks relevant information."
    )
]

# Initialize Groq LLM
try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.1,  # Lower temperature for more consistent medical responses
        max_tokens=2000
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
        handle_parsing_errors=True,
        max_iterations=3,  # Limit iterations to prevent long processing times
        early_stopping_method="generate"
    )
    logger.info("LangChain agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    raise

def parse_agent_output(text_data):
    """Parse agent output into structured JSON format"""
    result = {"diseases": [], "tests": []}
    
    # Handle case where agent couldn't provide medical advice
    if "cannot provide diagnostic test recommendations" in text_data.lower():
        result["tests"].append({
            "name": "Invalid Input",
            "description": "Cannot provide medical recommendations without valid symptoms, vitals, and age.",
            "tips": "Please provide complete and accurate medical information."
        })
        return result

    try:
        if "POSSIBLE DISEASES:" in text_data and "DIAGNOSTIC TESTS:" in text_data:
            # Split the text into diseases and tests sections
            parts = text_data.split("DIAGNOSTIC TESTS:")
            diseases_section = parts[0].replace("POSSIBLE DISEASES:", "").strip()
            tests_section = parts[1].strip()
            
            # Extract additional sections if present
            if "TIPS:" in tests_section:
                tests_section = tests_section.split("TIPS:")[0].strip()

            # Parse diseases
            for line in diseases_section.split('\n'):
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("•")):
                    # Remove bullet point and split on first " - "
                    clean_line = line.lstrip("- •").strip()
                    if " - " in clean_line:
                        parts = clean_line.split(" - ", 1)
                        if len(parts) == 2:
                            name, description = parts
                            result["diseases"].append({
                                "name": name.strip(),
                                "description": description.strip()
                            })

            # Parse tests
            for line in tests_section.split('\n'):
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("•")):
                    # Remove bullet point and split
                    clean_line = line.lstrip("- •").strip()
                    parts = clean_line.split(" - ")
                    
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        description = parts[1].strip() if len(parts) > 1 else "No description"
                        tips = parts[2].strip() if len(parts) > 2 else ""
                        
                        result["tests"].append({
                            "name": name,
                            "description": description,
                            "tips": tips
                        })
        
        # If no diseases or tests were parsed, add a fallback
        if not result["diseases"] and not result["tests"]:
            result["tests"].append({
                "name": "Parsing Error",
                "description": "Agent output format was not recognized.",
                "tips": "Please try again with clear symptoms and vitals."
            })
    
    except Exception as e:
        logger.error(f"Error parsing agent output: {e}")
        result["tests"].append({
            "name": "Processing Error",
            "description": "Error occurred while processing the medical analysis.",
            "tips": "Please try again."
        })

    return result

# Health check endpoint for AWS load balancer
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for load balancers"""
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        return jsonify({
            "status": "healthy", 
            "service": "medical-diagnostic-api",
            "mongodb": "connected",
            "timestamp": str(os.urandom(4).hex())
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "service": "medical-diagnostic-api", 
            "error": str(e)
        }), 503

# Main API endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    """Main endpoint for medical diagnostic recommendations"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400

        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid JSON payload"
            }), 400

        # Extract and validate input data
        symptoms = data.get("symptoms", "").strip()
        vitals = data.get("vitals", "").strip()
        age = data.get("age", "").strip()
        
        # Validate required fields
        if not symptoms or not vitals or not age:
            return jsonify({
                "success": False,
                "error": "Missing required fields: age, symptoms, or vitals."
            }), 400

        # Validate age is numeric
        try:
            age_num = int(age)
            if age_num < 0 or age_num > 150:
                return jsonify({
                    "success": False,
                    "error": "Age must be between 0 and 150."
                }), 400
        except ValueError:
            return jsonify({
                "success": False,
                "error": "Age must be a valid number."
            }), 400

        # Create the medical diagnostic prompt
        prompt = f"""
You are a medical assistant AI that provides diagnostic assistance. Based on the patient information provided, you must:

1. List possible diseases with brief descriptions that match the symptoms and vitals
2. Recommend diagnostic tests with justifications and suitability information

IMPORTANT RULES:
- If the input does NOT contain valid medical symptoms, vitals, or age, respond with:
  "I cannot provide diagnostic test recommendations or disease names without valid medical symptoms, vitals, and age."
- For valid medical inputs, use this EXACT format:

POSSIBLE DISEASES:
- Disease Name - Brief description based on symptoms and vitals.
- Disease Name - Brief description based on symptoms and vitals.

DIAGNOSTIC TESTS:
- Test Name - Clinical justification - Patient suitability notes.
- Test Name - Clinical justification - Patient suitability notes.

Patient Information:
Age: {age} years old
Symptoms: {symptoms}
Vitals: {vitals}

Provide evidence-based recommendations considering the patient's age and presentation.
"""

        logger.info(f"Processing diagnostic request for {age}-year-old patient")
        
        # Run the agent with timeout protection
        try:
            agent_output = agent.run(prompt)
            logger.info("Agent processing completed successfully")
        except Exception as agent_error:
            logger.error(f"Agent execution error: {agent_error}")
            return jsonify({
                "success": False,
                "error": "Unable to process medical analysis at this time. Please try again."
            }), 500

        # Parse the agent output
        parsed_output = parse_agent_output(agent_output)

        logger.info(f"Successfully processed diagnostic request: {len(parsed_output['diseases'])} diseases, {len(parsed_output['tests'])} tests")
        
        return jsonify({
            "success": True,
            "diseases": parsed_output["diseases"],
            "tests": parsed_output["tests"],
            "patient_age": age
        })

    except Exception as e:
        logger.error(f"Unexpected error in recommend endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": "An internal error occurred. Please try again later."
        }), 500

# Additional endpoint for testing connectivity
@app.route("/test-connections", methods=["GET"])
def test_connections():
    """Test endpoint to verify all service connections"""
    results = {}
    
    # Test MongoDB
    try:
        client.admin.command('ping')
        results["mongodb"] = "connected"
    except Exception as e:
        results["mongodb"] = f"failed: {str(e)}"
    
    # Test embedding model
    try:
        test_embedding = embedding_model.embed_query("test")
        results["embeddings"] = "working" if test_embedding else "failed"
    except Exception as e:
        results["embeddings"] = f"failed: {str(e)}"
    
    # Test Groq LLM
    try:
        test_response = llm.invoke("Say 'test successful'")
        results["groq_llm"] = "working" if test_response else "failed"
    except Exception as e:
        results["groq_llm"] = f"failed: {str(e)}"
    
    return jsonify({
        "service": "medical-diagnostic-api",
        "connections": results
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"success": False, "error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Health check available at: http://0.0.0.0:{port}/health")
    logger.info(f"API endpoint available at: http://0.0.0.0:{port}/recommend")
    
    # For production, use a proper WSGI server like Gunicorn
    app.run(host="0.0.0.0", port=port, debug=False)