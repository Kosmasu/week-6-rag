import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Updated imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# Define the models we'll use (all lightweight)
MODELS = ["llama3:8b", "gemma:7b", "mistral:7b"]

# Sample phishing-related data
phishing_data = [
    """Phishing is a type of social engineering attack often used to steal user data, including login credentials and credit card numbers. It occurs when an attacker, masquerading as a trusted entity, dupes a victim into opening an email, instant message, or text message. The recipient is then tricked into clicking a malicious link, which can lead to the installation of malware, the freezing of the system as part of a ransomware attack or the revealing of sensitive information.
    
    Common types of phishing attacks include:
    1. Email phishing: The most common type where attackers send fraudulent emails.
    2. Spear phishing: Targeted attacks against specific individuals or companies.
    3. Whaling: Attacks targeting senior executives and other high-profile targets.
    4. Smishing: Phishing conducted via SMS messages.
    5. Vishing: Phishing conducted via voice calls.""",
    
    """Indicators of phishing emails often include:
    1. Urgency: Creating a sense of urgency or fear to prompt immediate action.
    2. Poor spelling and grammar: Often a sign of unprofessional or fraudulent emails.
    3. Mismatched or suspicious URLs: The visible text might show a legitimate site, but hovering over it reveals a different URL.
    4. Generic greetings: Like "Dear Customer" instead of your name.
    5. Requests for personal information: Legitimate organizations typically don't ask for sensitive information via email.
    6. Suspicious attachments: These may contain malware.
    7. Too good to be true offers: If it sounds too good to be true, it probably is.""",
    
    """Real-world phishing example: The Gmail/Google Docs Worm (2017)
    This sophisticated attack sent victims an email from someone they knew, claiming they had shared a Google Doc. When users clicked the link and authorized what appeared to be Google Docs, they actually gave access to a third-party app named "Google Docs," which then harvested their contacts and forwarded the phishing email to everyone in their address book. The attack affected approximately one million Gmail users before Google shut it down within an hour.""",
    
    """Prevention techniques for phishing attacks include:
    1. Education and awareness training for users
    2. Email authentication protocols (SPF, DKIM, DMARC)
    3. Anti-phishing toolbars and browser extensions
    4. Multi-factor authentication (MFA)
    5. Regularly updated security software
    6. Email filtering at the gateway level
    7. Limited information sharing on social media
    8. Regular security assessments and penetration testing""",
    
    """Advanced phishing techniques:
    1. Clone phishing: Duplicating legitimate emails but replacing links with malicious ones
    2. Business Email Compromise (BEC): Targeting businesses that conduct wire transfers
    3. Session hijacking: Using compromised real website sessions to steal information
    4. DNS-based phishing: Modifying DNS settings to redirect users to fake websites
    5. Evil twin attacks: Creating rogue WiFi access points mimicking legitimate networks"""
]

test_queries = [
    "What are common indicators of phishing emails?",
    "How can organizations prevent phishing attacks?",
    "What was the Google Docs phishing attack?",
    "What is spear phishing and how does it differ from regular phishing?",
    "What techniques do phishers use to create urgency?"
]

# Ground truth answers for quality evaluation
ground_truth = [
    "Common indicators include urgency, poor spelling/grammar, suspicious URLs, generic greetings, requests for personal info, suspicious attachments, and too-good-to-be-true offers.",
    "Prevention techniques include education, email authentication, anti-phishing tools, multi-factor authentication, security software, email filtering, limited social media sharing, and regular security assessments.",
    "A 2017 attack where victims received emails appearing from contacts sharing Google Docs. When authorized, a third-party app harvested contacts and forwarded to others, affecting about one million Gmail users.",
    "Spear phishing targets specific individuals or organizations, unlike regular phishing which casts a wide net with generic messages to many potential victims.",
    "Phishers create urgency by using threatening language, imposing artificial deadlines, claiming security incidents, and using authoritative impersonation to force quick actions without verification."
]

def create_documents(texts: List[str]) -> List[Document]:
    """Convert text strings to Document objects."""
    return [Document(page_content=text) for text in texts]

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(model_name: str, documents: List[Document]):
    """Create a vector store using the specified embedding model."""
    embeddings = OllamaEmbeddings(model=model_name)
    return FAISS.from_documents(documents, embeddings)

def create_rag_chain(model_name: str, vectorstore):
    """Create a RAG chain using the specified language model and vector store."""
    llm = OllamaLLM(model=model_name)
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a cybersecurity expert specializing in phishing detection.
    Use the following pieces of context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """)
    
    # RAG pipeline
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Simplified chain structure
    retriever = vectorstore.as_retriever()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def measure_relevance(response: str, ground_truth: str) -> float:
    """
    Simple relevance score based on word overlap.
    In a real scenario, you might want to use a more sophisticated method.
    """
    response_words = set(response.lower().split())
    truth_words = set(ground_truth.lower().split())
    
    if not truth_words:
        return 0.0
    
    intersection = response_words.intersection(truth_words)
    return len(intersection) / len(truth_words)

def run_experiment():
    # Create and split documents
    print("Creating and splitting documents...")
    docs = create_documents(phishing_data)
    split_docs = split_documents(docs)
    
    results = []
    
    for model_name in MODELS:
        model_results = []
        print(f"\nTesting model: {model_name}")
        
        try:
            # Create vector store
            start_time = time.time()
            vectorstore = create_vectorstore(model_name, split_docs)
            indexing_time = time.time() - start_time
            print(f"Indexing time: {indexing_time:.2f} seconds")
            
            # Create RAG chain
            rag_chain = create_rag_chain(model_name, vectorstore)
            
            # Test queries
            for i, query in enumerate(test_queries):
                print(f"Processing query {i+1}: {query}")
                
                try:
                    # Measure response time
                    start_time = time.time()
                    response = rag_chain.invoke(query)  # Pass query string directly
                    query_time = time.time() - start_time
                    
                    # Calculate relevance score
                    relevance = measure_relevance(response, ground_truth[i])
                    
                    # Store results
                    model_results.append({
                        "model": model_name,
                        "query_id": i+1,
                        "query": query,
                        "response_time": query_time,
                        "relevance_score": relevance,
                        "response_length": len(response),
                        "response": response
                    })
                    
                    print(f"Response time: {query_time:.2f} seconds, Relevance: {relevance:.2f}")
                except Exception as e:
                    print(f"Error processing query {i+1}: {str(e)}")
                    model_results.append({
                        "model": model_name,
                        "query_id": i+1,
                        "query": query,
                        "response_time": 0,
                        "relevance_score": 0,
                        "response_length": 0,
                        "response": f"ERROR: {str(e)}"
                    })
            
            results.extend(model_results)
        except Exception as e:
            print(f"Error testing model {model_name}: {str(e)}")
    
    return pd.DataFrame(results)

def visualize_results(results_df):
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 12))
    
    # 1. Response time comparison
    plt.subplot(2, 2, 1)
    sns.barplot(x="model", y="response_time", data=results_df, palette="viridis")
    plt.title("Average Response Time by Model", fontsize=14)
    plt.ylabel("Time (seconds)")
    plt.xlabel("Model")
    
    # 2. Relevance score comparison
    plt.subplot(2, 2, 2)
    sns.barplot(x="model", y="relevance_score", data=results_df, palette="viridis")
    plt.title("Average Relevance Score by Model", fontsize=14)
    plt.ylabel("Relevance Score")
    plt.xlabel("Model")
    
    # 3. Response length comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x="model", y="response_length", data=results_df, palette="viridis")
    plt.title("Average Response Length by Model", fontsize=14)
    plt.ylabel("Length (characters)")
    plt.xlabel("Model")
    
    # 4. Relevance vs Response time scatter plot
    plt.subplot(2, 2, 4)
    sns.scatterplot(x="response_time", y="relevance_score", hue="model", 
                    size="response_length", sizes=(50, 200), data=results_df, palette="viridis")
    plt.title("Relevance vs Response Time", fontsize=14)
    plt.ylabel("Relevance Score")
    plt.xlabel("Response Time (seconds)")
    
    plt.tight_layout()
    plt.savefig("model_comparison_results.png", dpi=300)
    plt.show()
    
    # Query-specific performance
    plt.figure(figsize=(14, 8))
    pivoted = results_df.pivot(index="query_id", columns="model", values="relevance_score")
    sns.heatmap(pivoted, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Relevance Score by Query and Model", fontsize=14)
    plt.ylabel("Query ID")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig("query_performance_heatmap.png", dpi=300)
    plt.show()

def generate_report(results_df):
    # Generate HTML report
    html = """
    <html>
    <head>
        <title>Embedding Model Comparison for Cybersecurity RAG</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2, h3 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .summary { margin: 20px 0; padding: 20px; background-color: #f9f9f9; border-radius: 5px; }
            .model-section { margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
            .metrics { display: flex; justify-content: space-between; margin: 20px 0; }
            .metric-box { background-color: #f8f9fa; padding: 15px; border-radius: 5px; width: 30%; text-align: center; }
            .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
            .metric-label { font-size: 14px; color: #7f8c8d; }
            .query-box { margin: 15px 0; padding: 15px; border-radius: 5px; border: 1px solid #eee; }
            .response-box { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin-top: 10px; }
            img { max-width: 100%; height: auto; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Embedding Model Comparison for Cybersecurity RAG</h1>
        <p>This report compares different Ollama models for RAG applications in cybersecurity, specifically for phishing detection tasks.</p>
        
        <div class="summary">
            <h2>Summary of Findings</h2>
    """
    
    # Add summary metrics
    summary_df = results_df.groupby("model").agg({
        "response_time": "mean",
        "relevance_score": "mean",
        "response_length": "mean"
    }).reset_index()
    
    html += """
        <table>
            <tr>
                <th>Model</th>
                <th>Avg. Response Time (s)</th>
                <th>Avg. Relevance Score</th>
                <th>Avg. Response Length</th>
            </tr>
    """
    
    for _, row in summary_df.iterrows():
        html += f"""
            <tr>
                <td>{row['model']}</td>
                <td>{row['response_time']:.2f}</td>
                <td>{row['relevance_score']:.2f}</td>
                <td>{row['response_length']:.0f}</td>
            </tr>
        """
    
    html += """
        </table>
        </div>
        
        <h2>Visualizations</h2>
        <p>The following visualizations show the performance metrics of each model:</p>
        <img src="model_comparison_results.png" alt="Model Comparison Results">
        <img src="query_performance_heatmap.png" alt="Query Performance Heatmap">
        
        <h2>Detailed Model Performance</h2>
    """
    
    # Add detailed results for each model
    for model in MODELS:
        model_df = results_df[results_df["model"] == model]
        
        html += f"""
        <div class="model-section">
            <h3>Model: {model}</h3>
            
            <div class="metrics">
                <div class="metric-box">
                    <div class="metric-value">{model_df["response_time"].mean():.2f}s</div>
                    <div class="metric-label">Average Response Time</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{model_df["relevance_score"].mean():.2f}</div>
                    <div class="metric-label">Average Relevance Score</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{model_df["response_length"].mean():.0f}</div>
                    <div class="metric-label">Average Response Length</div>
                </div>
            </div>
            
            <h4>Query Results</h4>
        """
        
        for _, row in model_df.iterrows():
            html += f"""
            <div class="query-box">
                <h5>Query {row['query_id']}: {row['query']}</h5>
                <p><strong>Ground Truth:</strong> {ground_truth[int(row['query_id'])-1]}</p>
                <p><strong>Response Time:</strong> {row['response_time']:.2f}s | <strong>Relevance Score:</strong> {row['relevance_score']:.2f}</p>
                <div class="response-box">{row['response']}</div>
            </div>
            """
        
        html += "</div>"
    
    html += """
        <h2>Methodology</h2>
        <p>This experiment was conducted using the following methodology:</p>
        <ol>
            <li>Created a dataset of phishing-related information</li>
            <li>Split documents into chunks of 500 tokens with 50 token overlap</li>
            <li>Created embeddings using each model</li>
            <li>Built RAG systems using each model for both embeddings and generation</li>
            <li>Tested each system with the same set of cybersecurity queries</li>
            <li>Measured response time, relevance (compared to ground truth), and response length</li>
        </ol>
        
        <h2>Conclusion</h2>
        <p>Based on the results, we can observe which models perform best for cybersecurity RAG applications in terms of speed, relevance, and verbosity. This information can help security teams select the most appropriate model for their specific needs, balancing performance and resource requirements.</p>
    </body>
    </html>
    """
    
    with open("cybersecurity_rag_comparison.html", "w") as f:
        f.write(html)
    
    print("Generated HTML report: cybersecurity_rag_comparison.html")

def main():
    print("Starting experiment: Comparing Embedding Models for Cybersecurity RAG")
    
    # Run experiment
    results_df = run_experiment()
    
    # Save raw data
    results_df.to_csv("cybersecurity_rag_comparison_results.csv", index=False)
    print("Saved raw results to: cybersecurity_rag_comparison_results.csv")
    
    # Generate visualizations
    if not results_df.empty:
        print("Generating visualizations...")
        visualize_results(results_df)
        
        # Generate report
        print("Generating comprehensive report...")
        generate_report(results_df)
    else:
        print("No results to visualize - all models failed.")
    
    print("Experiment complete!")

if __name__ == "__main__":
    main()