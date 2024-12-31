import modal
from modal import App, Volume, Image
import base64
from typing import List, Optional, Dict
import tempfile
import os
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup infrastructure
app = modal.App("colpali_rag")
volume = Volume.from_name("model-cache", create_if_missing=True)
image = (
    Image.debian_slim()
    .apt_install("poppler-utils")
    .pip_install(
        "byaldi",
        "numpy",
        "IPython",
        "pdf2image",
        "pillow"
    )
)

secrets = [modal.Secret.from_name("huggingface-secret")]

# Constants
GPU = "A100"
MODEL = "vidore/colpali-v1.2-merged"
MODEL_DIR = "/cache/models"
PERSISTENT_CACHE_DIR = "/cache/hf_cache"

@app.cls(
    image=image,
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    volumes={"/cache": volume},
    keep_warm=2,
    container_idle_timeout=18000
)
class Companion:
    def __init__(self):
        self.model = None
        self.indexed_documents: Dict[str, bool] = {}
        self.temp_dir = None
    @modal.build()
    def build(self):
      """Download model during build"""
      from huggingface_hub import snapshot_download
      import os

      # Create directories
      os.makedirs(MODEL_DIR, exist_ok=True)
      os.makedirs(PERSISTENT_CACHE_DIR, exist_ok=True)

      # Set HuggingFace cache directory
      os.environ['HF_HOME'] = PERSISTENT_CACHE_DIR
    
      # Download model only if it doesn't exist
      model_path = os.path.join(MODEL_DIR, MODEL.split('/')[-1])
      if not os.path.exists(model_path):
          logger.info("Downloading model for the first time...")
          snapshot_download(MODEL, local_dir=model_path)
      else:
          logger.info("Model already exists in cache")

    @modal.enter()
    def enter(self):
      """Initialize model and temporary directory"""
      if self.model is None:
          from byaldi import RAGMultiModalModel
          import os

          # Set cache directory
          os.environ['HF_HOME'] = PERSISTENT_CACHE_DIR
        
          base_model_path = os.path.join(MODEL_DIR, MODEL.split('/')[-1])
          logger.info(f"Loading model from {base_model_path}")
        
          self.model = RAGMultiModalModel.from_pretrained(base_model_path)
          logger.info("Model loaded successfully")

      if self.temp_dir is None:
          self.temp_dir = tempfile.mkdtemp()
          logger.info(f"Created temporary directory: {self.temp_dir}")

    def process_single_page(self, image, page_num: int) -> Optional[str]:
        """Process a single page and return its file path if successful"""
        try:
            # Ensure we're working with RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as PNG
            path = os.path.join(self.temp_dir, f"page_{page_num}.png")
            image.save(path, format='PNG', optimize=True)
            
            if self.validate_image(path):
                logger.info(f"Successfully validated image for page {page_num}")
                return path
            return None
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            return None

    def process_pdf(self, pdf_bytes: bytes) -> List[str]:
        """Convert PDF bytes to a list of temporary image file paths"""
        from pdf2image import convert_from_bytes
        
        logger.info("Starting PDF conversion...")
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200)
            logger.info(f"Successfully converted {len(images)} pages from PDF")
            
            valid_paths = []
            for i, img in enumerate(images):
                path = self.process_single_page(img, i)
                if path:
                    valid_paths.append(path)
                    logger.info(f"Successfully processed page {i}")
            
            logger.info(f"Successfully processed {len(valid_paths)} pages out of {len(images)}")
            return valid_paths
            
        except Exception as e:
            logger.error(f"Error in PDF conversion: {str(e)}")
            return []

    def validate_image(self, image_path: str) -> bool:
        """Validate that the image file is properly formatted and readable"""
        try:
            from PIL import Image as PILImage
            with PILImage.open(image_path) as img:
                img.load()
                logger.info(f"Image validation successful - Path: {image_path}, Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
                return True
        except Exception as e:
            logger.error(f"Image validation failed for {image_path}: {str(e)}")
            return False

    def add_to_index(self, image_path: str, index_name: str) -> bool:
        """Safely add an image to the index"""
        try:
            logger.info(f"Starting indexing for {image_path}")
            
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return False

            try:
                # For the first page, create a new index
                self.model.index(image_path, index_name=index_name)
                logger.info(f"Created new index with {image_path}")
                return True
            except Exception as first_error:
                logger.info(f"Index exists, trying to add to it: {str(first_error)}")
                try:
                    # For subsequent pages, add to the existing index
                    self.model.add_to_index(image_path, store_collection_with_index=True)
                    logger.info(f"Successfully added {image_path} to existing index")
                    return True
                except Exception as second_error:
                    logger.error(f"Failed to add to index: {str(second_error)}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error in add_to_index: {str(e)}")
            return False

    @modal.method()
    def index_document(self, doc: bytes, doc_id: Optional[str] = None) -> str:
        """Index a document and return its index ID"""
        if self.model is None:
            raise RuntimeError("Model not initialized. This shouldn't happen!")
        
        # Generate or use provided document ID
        index_name = doc_id if doc_id else f"index_{int(time.time())}"
        
        image_paths = []
        try:
            # Process PDF into images
            logger.info("Starting PDF processing...")
            image_paths = self.process_pdf(doc)
            
            if not image_paths:
                raise ValueError("No images were extracted from the PDF")
            
            logger.info(f"Successfully extracted {len(image_paths)} images")
            
            # Index each page
            successful_indexes = 0
            for img_path in image_paths:
                if self.add_to_index(img_path, index_name):
                    successful_indexes += 1
                    time.sleep(1)
            
            if successful_indexes == 0:
                raise ValueError("No pages were successfully indexed")
            
            logger.info(f"Successfully indexed {successful_indexes} pages")
            
            # Mark document as indexed
            self.indexed_documents[index_name] = True
            
            return index_name
            
        except Exception as e:
            logger.error(f"Error in indexing: {str(e)}")
            raise
        finally:
            # Cleanup
            for path in image_paths:
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {str(e)}")

    @modal.method()
    def search(self, query: str, k: int = 1) -> List[bytes]:
        """Search across indexed documents"""
        if self.model is None:
            raise RuntimeError("Model not initialized. This shouldn't happen!")

        try:
            # Check if any documents have been indexed
            if not self.indexed_documents:
                raise ValueError("No documents have been indexed yet. Please index documents before searching.")

            logger.info(f"Performing search with query: {query}")
            results = self.model.search(query, k=k, return_base64_results=True)
            
            # Process results
            output_images = []
            for result in results:
                if hasattr(result, 'base64') and result.base64:
                    image_bytes = base64.b64decode(result.base64)
                    output_images.append(image_bytes)
            
            logger.info(f"Found {len(output_images)} matching results")
            return output_images
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            raise

    @modal.method()
    def get_indexed_documents(self) -> List[str]:
        """Get list of indexed document IDs"""
        return list(self.indexed_documents.keys())

    def __del__(self):
        try:
            if self.temp_dir:
                os.rmdir(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to remove temp directory: {str(e)}")