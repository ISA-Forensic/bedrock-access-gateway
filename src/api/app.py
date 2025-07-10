import logging
import sys
import os
import time

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Set up logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("=== BEDROCK ACCESS GATEWAY STARTUP ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path}")

try:
    logger.info("Importing routers...")
    from api.routers import chat, embeddings, model, knowledge_base, auth_ui
    logger.info("✅ All routers imported successfully")
except Exception as e:
    logger.error(f"❌ Error importing routers: {e}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

try:
    logger.info("Importing settings...")
    from api.setting import API_ROUTE_PREFIX, DESCRIPTION, SUMMARY, TITLE, VERSION
    logger.info(f"✅ Settings imported: API_ROUTE_PREFIX={API_ROUTE_PREFIX}")
except Exception as e:
    logger.error(f"❌ Error importing settings: {e}")
    raise

# Test schema imports
try:
    logger.info("Testing schema imports...")
    from api.schema import KnowledgeBase, KnowledgeBases
    logger.info("✅ Schema imports successful")
    
    # Test creating a KnowledgeBase object
    test_kb = KnowledgeBase(id="test", knowledge_base_id="test-id")
    logger.info(f"✅ Test KnowledgeBase created: {test_kb.model_dump()}")
    
    # Check for problematic models
    try:
        from api.schema import KnowledgeUserModel
        logger.error("❌ WARNING: KnowledgeUserModel found (should not exist!)")
    except ImportError:
        logger.info("✅ KnowledgeUserModel does not exist (correct)")
        
except Exception as e:
    logger.error(f"❌ Schema import/test failed: {e}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# Initialize monitoring
try:
    logger.info("Initializing monitoring...")
    from api.monitoring import metrics, start_metrics_server
    
    # Start Prometheus metrics server in a separate thread
    metrics_port = int(os.environ.get("METRICS_PORT", "8001"))
    if start_metrics_server(metrics_port):
        logger.info(f"✅ Metrics server started on port {metrics_port}")
    else:
        logger.warning("⚠️ Metrics server failed to start")
        
except Exception as e:
    logger.error(f"❌ Monitoring initialization failed: {e}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    # Continue without monitoring rather than crash

config = {
    "title": TITLE,
    "description": DESCRIPTION,
    "summary": SUMMARY,
    "version": VERSION,
}

logger.info(f"Creating FastAPI app with config: {config}")
app = FastAPI(**config)
logger.info("✅ FastAPI app created")

logger.info("Adding CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("✅ CORS middleware added")

# Add monitoring middleware
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Monitor all HTTP requests"""
    start_time = time.time()
    
    # Increment active connections
    try:
        metrics.increment_active_connections()
    except:
        pass  # Continue if monitoring fails
    
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics (skip metrics endpoint to avoid self-loops)
        if not request.url.path.endswith("/metrics"):
            try:
                metrics.record_api_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    duration=duration,
                    model="gateway"  # General gateway requests
                )
            except:
                pass  # Continue if monitoring fails
        
        return response
        
    except Exception as e:
        # Record error
        duration = time.time() - start_time
        status_code = getattr(e, 'status_code', 500)
        
        try:
            metrics.record_api_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                duration=duration,
                model="gateway"
            )
            metrics.record_error(
                error_type=type(e).__name__,
                endpoint=request.url.path,
                details=str(e)
            )
        except:
            pass  # Continue if monitoring fails
        
        raise
    finally:
        # Decrement active connections
        try:
            metrics.decrement_active_connections()
        except:
            pass

logger.info("✅ Monitoring middleware added")

logger.info("Including routers...")
app.include_router(model.router, prefix=API_ROUTE_PREFIX)
logger.info(f"✅ Model router included with prefix: {API_ROUTE_PREFIX}")

app.include_router(chat.router, prefix=API_ROUTE_PREFIX)
logger.info(f"✅ Chat router included with prefix: {API_ROUTE_PREFIX}")

app.include_router(embeddings.router, prefix=API_ROUTE_PREFIX)
logger.info(f"✅ Embeddings router included with prefix: {API_ROUTE_PREFIX}")

app.include_router(knowledge_base.router, prefix=API_ROUTE_PREFIX)
logger.info(f"✅ Knowledge base router included with prefix: {API_ROUTE_PREFIX}")

app.include_router(auth_ui.router, prefix=API_ROUTE_PREFIX)
logger.info(f"✅ Auth UI router included with prefix: {API_ROUTE_PREFIX}")

# Serve static simple UI
static_dir = Path(__file__).parent / "static"
logger.info(f"Setting up static files from: {static_dir}")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
logger.info("✅ Static files mounted")

@app.on_event("startup")
async def startup_event():
    """Log when the application starts up"""
    logger.info("🚀 BEDROCK ACCESS GATEWAY STARTUP COMPLETE")
    logger.info(f"API available at: http://localhost:8080{API_ROUTE_PREFIX}")
    logger.info(f"Health check: http://localhost:8080/health")
    logger.info(f"Metrics: http://localhost:8080/metrics")
    logger.info(f"Knowledge bases: http://localhost:8080{API_ROUTE_PREFIX}/knowledge-bases")
    
    # Test knowledge base manager on startup
    try:
        from api.kb_config_manager import get_kb_config_manager
        kb_manager = get_kb_config_manager()
        kbs = kb_manager.get_knowledge_bases()
        logger.info(f"✅ Knowledge base manager working, found {len(kbs)} knowledge bases")
    except Exception as e:
        logger.error(f"❌ Knowledge base manager test failed: {e}")
    
    # Test monitoring
    try:
        metrics.record_api_request("GET", "/startup-test", 200, 0.1, "system")
        logger.info("✅ Monitoring system working")
    except Exception as e:
        logger.error(f"❌ Monitoring test failed: {e}")

@app.get("/health")
async def health():
    """For health check if needed"""
    return {"status": "OK"}

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/debug/environment")
async def debug_environment():
    """Debug endpoint to check environment details"""
    try:
        import platform
        from api.kb_config_manager import get_kb_config_manager
        from api.schema import KnowledgeBase
        
        # Test creating a KnowledgeBase object
        test_kb = KnowledgeBase(id="debug-test", knowledge_base_id="test-id")
        
        # Get knowledge base manager info
        kb_manager = get_kb_config_manager()
        kbs = kb_manager.get_knowledge_bases()
        
        debug_info = {
            "status": "OK",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "working_directory": os.getcwd(),
            "api_route_prefix": API_ROUTE_PREFIX,
            "knowledge_base_manager": {
                "type": str(type(kb_manager)),
                "config_path": str(kb_manager.config_path),
                "config_exists": kb_manager.config_path.exists(),
                "knowledge_bases_count": len(kbs),
                "knowledge_bases": kbs
            },
            "schema_test": {
                "knowledge_base_created": test_kb.model_dump(),
                "knowledge_user_model_exists": False  # Should always be False
            },
            "environment_variables": {
                key: os.environ.get(key, "NOT_SET") 
                for key in ["API_ROUTE_PREFIX", "KB_CONFIG_PATH", "MODELS_CONFIG_PATH", "DEBUG"]
            }
        }
        
        # Test for problematic models
        try:
            from api.schema import KnowledgeUserModel
            debug_info["schema_test"]["knowledge_user_model_exists"] = True
            debug_info["schema_test"]["error"] = "KnowledgeUserModel found (should not exist!)"
        except ImportError:
            pass  # This is expected and correct
            
        return debug_info
        
    except Exception as e:
        import traceback
        return {
            "status": "ERROR",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
