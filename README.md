# Colpali RAG System

The Colpali RAG (Retrieval-Augmented Generation) system is a cloud-deployed solution that enables efficient document indexing and retrieval using the Colpali v1.2 language model, integrated with a Streamlit frontend and Llama 3.2 Vision for answer generation.

## System Architecture

### Backend:

- Uses vidore/colpali-v1.2-merged (4GB) from HuggingFace.
- Accessed via the byaldi library.
- Fully deployed on the Modal Cloud Platform.


### Frontend:

- Built with Streamlit
- Interactive chat interface
- Document upload functionality
- Real-time response
- Connected to the backend via APIs


### Answer Generation:

- Llama 3.2 Vision model
- Accessed via Groq API
- Processes retrieved images to generate answers


### Infrastructure:

- Deployed using Modal's serverless infrastructure
- Utilizes A100 GPU for computation
- Implements persistent volume storage for model caching

### Implementation Details

#### PDF Conversion:

- Converts PDF documents to images using pdf2image
- Processes at 200 DPI for optimal quality
- Converts images to RGB format for compatibility


#### Image Processing:

- Validates image format and readability
- Optimizes images using PNG format
- Implements error handling for corrupt or invalid images


#### Indexing System:

- Creates unique indices for documents
- Supports both new index creation and additions to existing indices
- Maintains document tracking through indexed_documents dictionary



#### Frontend Implementation
##### User Interface Components

Document Upload:

- Supports PDF file uploads
- Automatic document processing and indexing
- Progress feedback to users


Chat Interface:

- Interactive message history
- Real-time query input
- Styled message display with user/bot differentiation
- Session state management for conversation persistence


Document Management:

- Clear document functionality
- Session state tracking
- Error handling and user feedback

Integration with Llama 3.2 Vision
Query Processing Flow

Image Retrieval:

- Retrieves top-k relevant images from indexed documents
- Combines query text and retrieved images
- Base64 encoding for image data

Answer Generation:

- Uses Llama 3.2 Vision model via Groq
- Configurable generation parameters
- Structured response formatting