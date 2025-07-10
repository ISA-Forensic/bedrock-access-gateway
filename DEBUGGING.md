# Debugging Environment Issues

## Overview
This application now includes comprehensive logging and debugging tools to help identify environment-specific issues, particularly with knowledge base functionality.

## New Logging Features

### 1. Startup Logging
The application now logs detailed information during startup:
- Python version and environment details
- Module import status
- Schema validation tests
- Knowledge base manager initialization

### 2. Knowledge Base Endpoint Logging
All knowledge base endpoints now include detailed logging:
- Configuration loading process
- Individual knowledge base processing
- Object creation steps
- Error details with full stack traces

### 3. Debug Endpoint
A new debug endpoint provides environment information:
```
GET /debug/environment
```

## How to Use the Logging

### 1. Test Locally First
```bash
# Run the logging test script
python3 test_logging.py

# Start the application
cd src
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8080

# Check the startup logs in your terminal
```

### 2. Docker Environment
```bash
# Rebuild with no cache to ensure fresh environment
cd src
docker build --no-cache -t bedrock-access-gateway .

# Run with logging
docker run -d --name bedrock-gateway -p 8080:80 -e DEBUG=true bedrock-access-gateway

# Check logs
docker logs -f bedrock-gateway
```

### 3. Debug Endpoint Usage
Compare the debug output between environments:

**Local:**
```bash
curl http://localhost:8080/debug/environment | jq
```

**Remote:**
```bash
curl http://your-remote-server:8080/debug/environment | jq
```

### 4. Knowledge Base Endpoint Testing
Test the knowledge base endpoint with detailed logging:

```bash
# With authentication
curl -H "Authorization: Bearer bedrock" http://localhost:8080/api/v1/knowledge-bases

# Check logs immediately after the request
docker logs bedrock-gateway --tail 50
```

## Troubleshooting Steps

### 1. Compare Environment Debug Info
Run the debug endpoint on both environments and compare:
- Python versions
- Working directories
- Configuration file paths
- Environment variables
- Knowledge base counts

### 2. Check for Cached Code
Look for these indicators in the logs:
- "KnowledgeUserModel found (should not exist!)"
- Import errors mentioning old classes
- Traceback pointing to cached .pyc files

### 3. Verify Configuration Files
Check the logs for:
- Configuration file loading status
- File paths and existence
- JSON parsing errors
- Knowledge base counts

### 4. Clean Environment Setup
If issues persist:

```bash
# Complete cleanup
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker rmi $(docker images -q)
docker system prune -a -f --volumes

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Rebuild from scratch
cd src
docker build --no-cache --pull -t bedrock-access-gateway .
```

## Log Analysis

### Key Log Markers
Look for these markers in the logs:

**Success Markers:**
- `✅ All routers imported successfully`
- `✅ Schema imports successful`
- `✅ KnowledgeUserModel does not exist (correct)`
- `=== LIST KNOWLEDGE BASES REQUEST SUCCESS ===`

**Error Markers:**
- `❌ Error importing routers`
- `❌ WARNING: KnowledgeUserModel found`
- `=== LIST KNOWLEDGE BASES REQUEST FAILED ===`
- `Error creating KnowledgeBase object`

### Common Issues

**1. Cached Old Code:**
```
❌ WARNING: KnowledgeUserModel found (should not exist!)
```
Solution: Complete Docker cleanup and rebuild

**2. Configuration File Issues:**
```
Knowledge base config file not found at /path/to/config
```
Solution: Check file paths and Docker volume mounts

**3. Import Errors:**
```
❌ Error importing routers: ModuleNotFoundError
```
Solution: Check Python path and package installation

## Environment Comparison Checklist

When comparing local vs remote, check:

- [ ] Same Python version
- [ ] Same working directory structure
- [ ] Same configuration file contents
- [ ] Same environment variables
- [ ] No KnowledgeUserModel in schema
- [ ] Knowledge base manager working
- [ ] No cached .pyc files
- [ ] Docker image built with --no-cache

## Quick Debug Commands

```bash
# Check if app is responding
curl -f http://localhost:8080/health

# Get detailed environment info
curl http://localhost:8080/debug/environment | jq '.knowledge_base_manager'

# Test knowledge bases endpoint
curl -H "Authorization: Bearer bedrock" http://localhost:8080/api/v1/knowledge-bases

# Check recent logs
docker logs bedrock-gateway --tail 100 | grep -E "(✅|❌|===)"
```

## Next Steps

If you're still experiencing issues after using these debugging tools:

1. Share the output of `/debug/environment` from both environments
2. Share the startup logs showing the import and initialization process
3. Share the specific error logs when accessing `/knowledge-bases`
4. Verify that the source code is identical in both environments 