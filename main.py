import sys
import threading
import time
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtWidgets import QMessageBox
# Enable Qt integration with Jupyter

# First, make sure the UI file is converted to Python

from fusebot_v6_ui import Ui_FuseBot

# Import LLM components
import litellm
from langchain_litellm import ChatLiteLLM
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Import document processing components
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import memory components
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
import tiktoken

# LiteLLM Proxy Configuration
PROXY_API_BASE = "https://llmgateway.itg.ti.com"
PROXY_API_KEY = "sk-o5FfSmOUBYbt_-o9TCCksg"  # Replace with your actual LiteLLM Proxy key
litellm.use_litellm_proxy = True

# Available Models Configuration
AVAILABLE_MODELS = {
    "Gemini-2.5-pro": {
        "model_id": "vertex_ai/gemini-2.5-pro",
        "description": "Google's advanced multimodal model",
        "context_length": 1000000  # 1M tokens
    },
    "Gemini-2.5-flash": {
        "model_id": "vertex_ai/gemini-2.5-flash",
        "description": "gemini flash model",
        "context_length": 1000000  # 1M tokens
    },
    "Claude-3-7-sonnet": {
        "model_id": "anthropic.claude-3-7-sonnet-ep",
        "description": "Claude sonnet model",
        "context_length": 200000  # 200K tokens
    },
    "Claude-3-5-haiku": {
        "model_id": "anthropic.claude-3-5-haiku-ep",
        "description": "Calude model",
        "context_length": 200000  # 200K tokens
    },
    "Llama-3.3-70B": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "description": "LLama model",
        "context_length": 128000  # 128K tokens
    }
}

# Create a signals class for thread-safe communication
class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(str)
    remove_thinking = pyqtSignal()
    file_loaded = pyqtSignal(str, str)  # Signal for file loaded (filename, content)

class FuseBotApp(QMainWindow):
    def __init__(self):
        super(FuseBotApp, self).__init__()
        self.ui = Ui_FuseBot()
        self.ui.setupUi(self)
        
        # Initialize signals
        self.signals = WorkerSignals()
        self.signals.result.connect(self.display_bot_response)
        self.signals.error.connect(self.display_error)
        self.signals.remove_thinking.connect(self.remove_thinking)
        self.signals.file_loaded.connect(self.on_file_loaded)
        
        # Initialize chat components
        self.is_processing = False
        self.llm = None
        self.memory = None
        self.conversation_chain = None
        self.file_content = None
        self.file_filename = None
        self.system_prompt = (
            "You are a helpful and friendly chatbot. "
            "Format your answers using Markdown. "
            "Use headers for titles, bold for emphasis, and properly formatted code blocks for code snippets."
        )
        
        # Connect UI elements to functions
        self.ui.sendButton.clicked.connect(self.send_message)
        self.ui.messageInput.installEventFilter(self)  # For Enter key press
        self.ui.modelSelector.currentIndexChanged.connect(self.initialize_model)
        
        # Connect file upload button
        self.ui.fileUpload.clicked.connect(self.upload_file)
        
        # Set window title
        self.setWindowTitle("FuseBot - AI Assistant")
        
        # Welcome message
        self.ui.chatHistoryBrowser.append("<b>FuseBot:</b> Hello! I'm FuseBot. How can I help you today?")
        
        # Initialize the first model
        self.initialize_model()
    
    def eventFilter(self, obj, event):
        # Handle Enter key press in message input
        if obj is self.ui.messageInput and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == Qt.Key_Return and not (event.modifiers() & Qt.ShiftModifier):
                self.send_message()
                return True
        return super().eventFilter(obj, event)
    
    def initialize_model(self):
        # Get selected model
        model_name = self.ui.modelSelector.currentText()
        if model_name not in AVAILABLE_MODELS:
            self.ui.chatHistoryBrowser.append("<b>System:</b> Error: Invalid model selection.")
            return
            
        selected_model = AVAILABLE_MODELS[model_name]
        
        # Setup LangChain components
        try:
            # 1. Set up the LangChain LLM
            self.llm = ChatLiteLLM(
                model=selected_model["model_id"],
                api_base=PROXY_API_BASE,
                api_key=PROXY_API_KEY,
                timeout=600
            )
            
            # 2. Create a separate, lightweight LLM for memory summarization
            llm_for_memory = ChatOpenAI(model_name="gpt-3.5-turbo", api_key="dummy-key-not-used")
            
            # 3. Set up the Conversation Summary Buffer Memory with 50% of the model's context length
            memory_token_limit = int(selected_model["context_length"] * 0.5)
            self.memory = ConversationSummaryBufferMemory(
                llm=llm_for_memory,
                max_token_limit=memory_token_limit,
                memory_key="chat_history",
                return_messages=True
            )
            
            # 4. Create the Prompt Template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{human_input}")
            ])
            
            # 5. Create the LLMChain
            self.conversation_chain = LLMChain(
                llm=self.llm,
                prompt=prompt_template,
                memory=self.memory,
                verbose=False
            )
            
            # Add model change notification to chat history
            self.ui.chatHistoryBrowser.append(f"<b>System:</b> Switched to model: {model_name}")
            self.ui.chatHistoryBrowser.append(f"<b>System:</b> Memory buffer set to {memory_token_limit} tokens (50% of context length)")
            
        except Exception as e:
            import traceback
            error_msg = f"Error initializing model: {str(e)}\n{traceback.format_exc()}"
            self.ui.chatHistoryBrowser.append(f"<b>System Error:</b> {error_msg}")
    
    def upload_file(self):
        """Handle file upload (PDF, CSV, TXT, XLS/XLSX, DOC/DOCX, PPT/PPTX)"""
        if self.is_processing:
            QMessageBox.warning(self, "Processing in Progress", "Please wait until the current operation completes.")
            return
            
        # Open file dialog to select supported files
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", 
            "Document Files (*.pdf *.csv *.txt *.xls *.xlsx *.doc *.docx *.ppt *.pptx)"
        )
        
        if not file_path:
            return  # User canceled
            
        # Show loading indicator
        self.ui.chatHistoryBrowser.append("<i>Loading file, please wait...</i>")
        self.is_processing = True
        
        # Process file in a separate thread
        threading.Thread(
            target=self.process_file,
            args=(file_path,),
            daemon=True
        ).start()
    
    def process_file(self, file_path):
        """Process the uploaded file in a separate thread"""
        try:
            # Get the filename and extension
            filename = os.path.basename(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                # Use PyPDFLoader to extract text from PDF
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                # Extract text from all pages
                full_text = ""
                for page in pages:
                    full_text += page.page_content + "\n\n"
                    
            elif file_extension == '.csv':
                # Use CSVLoader to extract data from CSV
                loader = CSVLoader(file_path)
                documents = loader.load()
                # Convert CSV data to readable text format
                full_text = f"CSV Data from {filename}:\n\n"
                # If we have documents, get the first one to extract headers
                if documents:
                    # Extract headers from metadata if available
                    if hasattr(documents[0], 'metadata') and 'column_names' in documents[0].metadata:
                        headers = documents[0].metadata['column_names']
                        full_text += "Headers: " + ", ".join(headers) + "\n\n"
                # Add each row's content
                for i, doc in enumerate(documents):
                    full_text += f"Row {i+1}:\n{doc.page_content}\n\n"
                    
            elif file_extension == '.txt':
                # Use TextLoader for plain text files
                loader = TextLoader(file_path)
                documents = loader.load()
                full_text = ""
                for doc in documents:
                    full_text += doc.page_content + "\n\n"
                    
            elif file_extension in ['.xls', '.xlsx']:
                # Use UnstructuredExcelLoader for Excel files
                loader = UnstructuredExcelLoader(file_path)
                documents = loader.load()
                full_text = f"Excel Data from {filename}:\n\n"
                for i, doc in enumerate(documents):
                    full_text += f"Section {i+1}:\n{doc.page_content}\n\n"
                    
            elif file_extension in ['.doc', '.docx']:
                # Try Docx2txtLoader first for .docx, fall back to UnstructuredWordDocumentLoader
                try:
                    if file_extension == '.docx':
                        loader = Docx2txtLoader(file_path)
                    else:
                        loader = UnstructuredWordDocumentLoader(file_path)
                    documents = loader.load()
                    full_text = f"Word Document Content from {filename}:\n\n"
                    for doc in documents:
                        full_text += doc.page_content + "\n\n"
                except Exception as e:
                    # Fallback to UnstructuredWordDocumentLoader if Docx2txtLoader fails
                    loader = UnstructuredWordDocumentLoader(file_path)
                    documents = loader.load()
                    full_text = f"Word Document Content from {filename}:\n\n"
                    for doc in documents:
                        full_text += doc.page_content + "\n\n"
                        
            elif file_extension in ['.ppt', '.pptx']:
                # Use UnstructuredPowerPointLoader for PowerPoint files
                loader = UnstructuredPowerPointLoader(file_path)
                documents = loader.load()
                full_text = f"PowerPoint Content from {filename}:\n\n"
                for i, doc in enumerate(documents):
                    full_text += f"Slide/Section {i+1}:\n{doc.page_content}\n\n"
                    
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
            # Signal that file is loaded
            self.signals.file_loaded.emit(filename, full_text)
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing file: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
        finally:
            self.is_processing = False
    
    @pyqtSlot(str, str)
    def on_file_loaded(self, filename, content):
        """Handle successful file loading"""
        # Store the content for later use
        self.file_filename = filename
        self.file_content = content
        
        # Calculate approximate token count
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(content))
        except:
            # Fallback to rough estimation
            token_count = len(content) // 4
            
        # Remove loading indicator (if any)
        html = self.ui.chatHistoryBrowser.toHtml()
        html = html.replace("<i>Loading file, please wait...</i>", "")
        self.ui.chatHistoryBrowser.setHtml(html)
        
        # Determine file type from extension
        file_extension = os.path.splitext(filename)[1].lower()
        file_type_map = {
            '.pdf': 'PDF',
            '.csv': 'CSV',
            '.txt': 'Text',
            '.xls': 'Excel',
            '.xlsx': 'Excel',
            '.doc': 'Word',
            '.docx': 'Word',
            '.ppt': 'PowerPoint',
            '.pptx': 'PowerPoint'
        }
        file_type = file_type_map.get(file_extension, "Document")
        
        # Add success message
        self.ui.chatHistoryBrowser.append(f"<b>System:</b> {file_type} '{filename}' loaded successfully. (~{token_count} tokens)")
        self.ui.chatHistoryBrowser.append("<b>System:</b> You can now ask questions about the document.")
    
    def send_message(self):
        # Get user input
        user_input = self.ui.messageInput.toPlainText().strip()
        if not user_input or self.is_processing:
            return
            
        # Clear the input field
        self.ui.messageInput.clear()
        
        # Display user message in chat history
        self.ui.chatHistoryBrowser.append(f"<b>You:</b> {user_input}")
        
        # Add thinking indicator
        self.ui.chatHistoryBrowser.append("<i>FuseBot is thinking...</i>")
        
        # Process the message in a separate thread to avoid UI freezing
        self.is_processing = True
        
        # Start processing thread
        threading.Thread(
            target=self.process_message,
            args=(user_input,),
            daemon=True
        ).start()
    
    def process_message(self, user_input):
        try:
            # Check if model is initialized
            if self.llm is None or self.conversation_chain is None:
                self.signals.remove_thinking.emit()
                self.signals.error.emit("Please select a model first.")
                self.is_processing = False
                return
                
            # If a file is loaded, include it in the context
            if hasattr(self, 'file_content') and self.file_content:
                # Determine file type from extension
                file_extension = os.path.splitext(self.file_filename)[1].lower()
                file_type_map = {
                    '.pdf': 'PDF',
                    '.csv': 'CSV',
                    '.txt': 'Text',
                    '.xls': 'Excel',
                    '.xlsx': 'Excel',
                    '.doc': 'Word',
                    '.docx': 'Word',
                    '.ppt': 'PowerPoint',
                    '.pptx': 'PowerPoint'
                }
                file_type = file_type_map.get(file_extension, "Document")
                
                # Modify the input to include context from the file
                enhanced_input = f"""
Question/Request: {user_input}
Context from {file_type} document '{self.file_filename}':
{self.file_content}
Please respond to my question/request using the information in the {file_type} document when relevant.
"""
            else:
                enhanced_input = user_input
                
            # Process the message using the conversation chain
            response = self.conversation_chain.invoke({"human_input": enhanced_input})
            
            # Get the response text
            full_response = response.get('text', '')
            
            # Signal to remove thinking message
            self.signals.remove_thinking.emit()
            
            # Send the result to the main thread
            self.signals.result.emit(self.markdown_to_html(full_response))
            
            # Estimate memory usage and display
            try:
                memory_data = self.memory.load_memory_variables({})
                memory_messages = memory_data.get("chat_history", [])
                memory_content = ""
                for msg in memory_messages:
                    memory_content += f"{msg.type}: {msg.content}\n"
                    
                # Estimate token count using tiktoken
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    token_count = len(encoding.encode(memory_content))
                except:
                    # Fallback to rough estimation
                    token_count = len(memory_content) // 4
                    
                # Get max token limit for current model
                model_name = self.ui.modelSelector.currentText()
                max_tokens = int(AVAILABLE_MODELS[model_name]["context_length"] * 0.5)
                
                # Display memory usage
                self.ui.chatHistoryBrowser.append(f"<i>Memory usage: ~{token_count}/{max_tokens} tokens</i>")
            except Exception as e:
                # If memory estimation fails, don't interrupt the conversation
                pass
                
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.signals.remove_thinking.emit()
            self.signals.error.emit(error_msg)
        finally:
            self.is_processing = False
    
    @pyqtSlot()
    def remove_thinking(self):
        html = self.ui.chatHistoryBrowser.toHtml()
        html = html.replace("<i>FuseBot is thinking...</i>", "")
        self.ui.chatHistoryBrowser.setHtml(html)
    
    @pyqtSlot(str)
    def display_bot_response(self, message):
        self.ui.chatHistoryBrowser.append(f"<b>FuseBot:</b> {message}")
    
    @pyqtSlot(str)
    def display_error(self, error_message):
        self.ui.chatHistoryBrowser.append(f"<b>Error:</b> {error_message}")
    
    def markdown_to_html(self, markdown_text):
        """Convert markdown to HTML using the markdown library"""
        import markdown
        # Convert the markdown to HTML
        html = markdown.markdown(markdown_text, extensions=['fenced_code', 'tables'])
        # Add some basic styling for code blocks if present
        html = html.replace(
            '<pre>',
            '<pre style="background-color: #1e1e1e; color: #d4d4d4; padding: 12px; '
            'border-radius: 4px; font-family: Consolas, \'Courier New\', monospace;">'
        )
        return html

# Create and show the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FuseBotApp()
    window.show()
    sys.exit(app.exec_())